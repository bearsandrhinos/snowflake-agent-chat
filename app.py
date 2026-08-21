"""Streamlit chat UI for the Cortex Agent in ANALYTICS_PROD.SAAS.OPPORTUNITIES."""

from __future__ import annotations

import json
from typing import Any, Optional

import streamlit as st

from agent_client import (
    AgentConfig,
    CortexAgentClient,
    iter_sse,
    result_set_to_frame,
)

st.set_page_config(page_title="Opportunities Agent", page_icon="💬", layout="wide")

VISIBLE_TYPES = frozenset({"text", "table", "chart"})


def init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "thread_id": None,
        "parent_message_id": 0,
        "client": None,
        "connected": False,
        "connect_error": None,
        "auto_connect_attempted": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_client() -> CortexAgentClient:
    client: Optional[CortexAgentClient] = st.session_state.client
    if client is None:
        raise RuntimeError("Not connected to Snowflake.")
    return client


def set_client(client: CortexAgentClient) -> None:
    old = st.session_state.client
    if old is not None and old is not client:
        try:
            old.close()
        except Exception:
            pass
    st.session_state.client = client
    st.session_state.connected = True
    st.session_state.connect_error = None
    st.session_state.thread_id = None
    st.session_state.parent_message_id = 0


def clear_client() -> None:
    if st.session_state.client is not None:
        try:
            st.session_state.client.close()
        except Exception:
            pass
    st.session_state.client = None
    st.session_state.connected = False


def ensure_key_pair_connection() -> None:
    """Auto-connect when Streamlit secrets / env provide a private key."""
    config = AgentConfig()
    if not config.uses_key_pair:
        return
    if st.session_state.connected and st.session_state.client is not None:
        return
    if st.session_state.auto_connect_attempted and st.session_state.connect_error:
        return

    st.session_state.auto_connect_attempted = True
    try:
        client = CortexAgentClient(config)
        client.connect()
        set_client(client)
    except Exception as exc:
        clear_client()
        st.session_state.connect_error = str(exc)


def visible_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only user-facing answer pieces (text / table / chart)."""
    return [item for item in content if item.get("type") in VISIBLE_TYPES]


def render_content_item(item: dict[str, Any]) -> None:
    item_type = item.get("type")
    if item_type == "text":
        text = item.get("text") or ""
        if text.strip():
            st.markdown(text)
    elif item_type == "table":
        table = item.get("table") or item
        title = table.get("title")
        if title:
            st.caption(title)
        result_set = table.get("result_set") or table.get("resultSet") or {}
        st.dataframe(result_set_to_frame(result_set), use_container_width=True)
    elif item_type == "chart":
        chart = item.get("chart") or item
        spec_raw = chart.get("chart_spec") or chart.get("chartSpec") or "{}"
        spec = json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
        st.vega_lite_chart(spec, use_container_width=True)


def render_debug(debug: dict[str, Any]) -> None:
    statuses = debug.get("statuses") or []
    thinking = (debug.get("thinking") or "").strip()
    tools = debug.get("tools") or []

    if not statuses and not thinking and not tools:
        return

    with st.expander("Debug details", expanded=False):
        if statuses:
            st.markdown("**Status**")
            for status in statuses:
                st.caption(f"• {status}")

        if thinking:
            st.markdown("**Thinking**")
            st.markdown(thinking)

        if tools:
            st.markdown("**Tools**")
            for i, tool in enumerate(tools, start=1):
                label = tool.get("name") or tool.get("type") or f"Step {i}"
                with st.expander(f"{i}. {label}", expanded=False):
                    st.json(tool)


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        for item in visible_content(message.get("content") or []):
            render_content_item(item)
        if role == "assistant":
            render_debug(message.get("debug") or {})


def collect_agent_response(prompt: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the agent; return visible content plus collapsed debug details."""
    client = get_client()
    if st.session_state.thread_id is None:
        st.session_state.thread_id = client.create_thread()

    response = client.run_agent(
        prompt,
        thread_id=st.session_state.thread_id,
        parent_message_id=st.session_state.parent_message_id,
    )

    final_content: list[dict[str, Any]] = []
    statuses: list[str] = []
    thinking_parts: list[str] = []
    tools: list[dict[str, Any]] = []

    for event_name, data in iter_sse(response):
        if event_name == "response.status":
            message = data.get("message") if isinstance(data, dict) else str(data)
            if message and (not statuses or statuses[-1] != message):
                statuses.append(message)
        elif event_name == "response.thinking.delta":
            text = data.get("text", "") if isinstance(data, dict) else ""
            if text:
                thinking_parts.append(text)
        elif event_name == "response.thinking":
            text = data.get("text", "") if isinstance(data, dict) else ""
            if text:
                thinking_parts = [text]
        elif event_name == "response.tool_use":
            if isinstance(data, dict):
                tools.append(
                    {
                        "kind": "use",
                        "name": data.get("name"),
                        "type": data.get("type"),
                        "data": data,
                    }
                )
        elif event_name == "response.tool_result":
            if isinstance(data, dict):
                tools.append(
                    {
                        "kind": "result",
                        "name": data.get("name") or data.get("type") or "tool_result",
                        "type": data.get("type"),
                        "data": data,
                    }
                )
        elif event_name == "metadata":
            meta = data.get("metadata", data) if isinstance(data, dict) else {}
            message_id = meta.get("message_id") or meta.get("id")
            if message_id is not None:
                st.session_state.parent_message_id = int(message_id)
        elif event_name == "error":
            code = data.get("code", "") if isinstance(data, dict) else ""
            message = (
                data.get("message", str(data)) if isinstance(data, dict) else str(data)
            )
            raise RuntimeError(
                f"Agent error{f' ({code})' if code else ''}: {message}"
            )
        elif event_name == "response":
            final_content = data.get("content") or []
            message_id = data.get("id") or data.get("message_id")
            if message_id is not None:
                st.session_state.parent_message_id = int(message_id)

            if not thinking_parts:
                for item in final_content:
                    if item.get("type") == "thinking":
                        thinking = item.get("thinking") or item
                        text = (
                            thinking.get("text", "")
                            if isinstance(thinking, dict)
                            else str(thinking)
                        )
                        if text.strip():
                            thinking_parts.append(text)

    debug = {
        "statuses": statuses,
        "thinking": "".join(thinking_parts).strip(),
        "tools": tools,
    }
    return visible_content(final_content), debug


def sidebar() -> None:
    config = AgentConfig()
    with st.sidebar:
        st.header("Connection")
        st.caption(f"`{config.database}.{config.schema}.{config.agent}`")

        if config.uses_key_pair:
            st.caption("Auth: key-pair (Streamlit secrets)")
            if st.session_state.connect_error:
                st.error(f"Connection failed: {st.session_state.connect_error}")
                if st.button("Retry connect", use_container_width=True):
                    st.session_state.auto_connect_attempted = False
                    st.session_state.connect_error = None
                    st.rerun()
        else:
            st.caption("Auth: browser SSO")
            user = st.text_input("Email", value=config.user or "")
            col1, col2 = st.columns(2)
            with col1:
                connect = st.button("Connect", type="primary", use_container_width=True)
            with col2:
                disconnect = st.button("Disconnect", use_container_width=True)

            if connect:
                if not user.strip():
                    st.error("Email is required.")
                else:
                    client = CortexAgentClient(
                        AgentConfig(
                            account=config.account,
                            user=user.strip(),
                            warehouse=config.warehouse,
                            database=config.database,
                            schema=config.schema,
                            agent=config.agent,
                            authenticator="externalbrowser",
                            private_key="",
                        )
                    )
                    try:
                        with st.spinner("Opening browser for Snowflake SSO…"):
                            client.connect()
                        set_client(client)
                        st.success("Connected")
                    except Exception as exc:
                        clear_client()
                        st.error(f"Connection failed: {exc}")

            if disconnect:
                clear_client()
                st.info("Disconnected")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.session_state.parent_message_id = 0
            st.rerun()

        st.divider()
        st.markdown(
            f"**Status:** {'Connected' if st.session_state.connected else 'Not connected'}"
        )
        if st.session_state.thread_id is not None:
            st.caption(f"Thread `{st.session_state.thread_id}`")


def main() -> None:
    init_state()
    ensure_key_pair_connection()
    sidebar()

    st.title("Opportunities Agent")
    st.caption("Ask questions — answers may include text, tables, or charts.")

    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input(
        "Ask about opportunities…",
        disabled=not st.session_state.connected,
    )
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    )

    with st.spinner("Asking the agent…"):
        try:
            content, debug = collect_agent_response(prompt)
        except Exception as exc:
            st.session_state.messages.pop()
            st.error(f"Request failed: {exc}")
            return

    if not content:
        st.session_state.messages.pop()
        st.info("No response content returned.")
        return

    st.session_state.messages.append(
        {"role": "assistant", "content": content, "debug": debug}
    )
    st.rerun()


if __name__ == "__main__":
    main()
