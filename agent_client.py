"""Snowflake connection and Cortex Agent REST client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import requests
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv

load_dotenv()


def _secret_section() -> dict[str, Any]:
    """Return [snowflake] from Streamlit secrets when available."""
    try:
        import streamlit as st

        section = st.secrets.get("snowflake", {})
        return dict(section) if section else {}
    except Exception:
        return {}


def _cfg(name: str, default: str = "") -> str:
    secrets = _secret_section()
    if name in secrets and secrets[name] not in (None, ""):
        return str(secrets[name]).strip()
    return os.getenv(f"SNOWFLAKE_{name.upper()}", default).strip()


def _private_key_pem() -> str:
    secrets = _secret_section()
    if secrets.get("private_key"):
        return str(secrets["private_key"]).strip()
    path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "").strip()
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    pem = os.getenv("SNOWFLAKE_PRIVATE_KEY", "").strip()
    return pem


def _load_private_key_bytes(pem: str) -> bytes:
    # Streamlit secrets / .env sometimes preserve literal "\n"
    normalized = pem.replace("\\n", "\n").strip()
    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
    password = passphrase.encode() if passphrase else None
    key = serialization.load_pem_private_key(
        normalized.encode("utf-8"),
        password=password,
        backend=default_backend(),
    )
    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


@dataclass
class AgentConfig:
    account: str = field(default_factory=lambda: _cfg("account", "GQB59211"))
    user: str = field(default_factory=lambda: _cfg("user"))
    warehouse: str = field(default_factory=lambda: _cfg("warehouse", "COMPUTE_WH"))
    database: str = field(default_factory=lambda: _cfg("database", "ANALYTICS_PROD"))
    schema: str = field(default_factory=lambda: _cfg("schema", "SAAS"))
    agent: str = field(default_factory=lambda: _cfg("agent", "OPPORTUNITIES"))
    authenticator: str = field(
        default_factory=lambda: _cfg("authenticator", "externalbrowser")
    )
    private_key: str = field(default_factory=_private_key_pem)

    @property
    def uses_key_pair(self) -> bool:
        return bool(self.private_key.strip())


class CortexAgentClient:
    """Authenticate to Snowflake and call the Cortex Agents API."""

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self._conn: Optional[snowflake.connector.SnowflakeConnection] = None

    def connect(self) -> None:
        if not self.config.user:
            raise ValueError(
                "Snowflake user is required. Set it in Streamlit secrets, .env, "
                "or the sidebar."
            )

        kwargs: dict[str, Any] = {
            "account": self.config.account,
            "user": self.config.user,
            "warehouse": self.config.warehouse,
            "database": self.config.database,
            "schema": self.config.schema,
            "client_session_keep_alive": True,
        }

        if self.config.uses_key_pair:
            kwargs["private_key"] = _load_private_key_bytes(self.config.private_key)
        else:
            kwargs["authenticator"] = self.config.authenticator or "externalbrowser"

        self._conn = snowflake.connector.connect(**kwargs)

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed()

    def close(self) -> None:
        if self._conn is not None and not self._conn.is_closed():
            self._conn.close()
        self._conn = None

    def _host(self) -> str:
        assert self._conn is not None
        return self._conn.host

    def _token(self) -> str:
        assert self._conn is not None
        return self._conn.rest.token  # type: ignore[attr-defined]

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f'Snowflake Token="{self._token()}"',
            "Content-Type": "application/json",
            "Accept": accept,
        }

    def create_thread(self) -> int:
        url = f"https://{self._host()}/api/v2/cortex/threads"
        resp = requests.post(url, headers=self._headers(), json={}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        thread_id = data.get("thread_id") or data.get("id")
        if thread_id is None:
            raise RuntimeError(f"Unexpected thread response: {data}")
        return int(thread_id)

    def run_agent(
        self,
        prompt: str,
        *,
        thread_id: Optional[int] = None,
        parent_message_id: int = 0,
    ) -> requests.Response:
        cfg = self.config
        url = (
            f"https://{self._host()}/api/v2/databases/{cfg.database}"
            f"/schemas/{cfg.schema}/agents/{cfg.agent}:run"
        )
        body: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            "stream": True,
        }
        if thread_id is not None:
            body["thread_id"] = thread_id
            body["parent_message_id"] = parent_message_id

        resp = requests.post(
            url,
            headers=self._headers(accept="text/event-stream"),
            data=json.dumps(body),
            stream=True,
            timeout=900,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Agent run failed ({resp.status_code}): {resp.text[:2000]}"
            )
        return resp


def iter_sse(response: requests.Response) -> Generator[tuple[str, Any], None, None]:
    """Yield (event_name, data) pairs from an SSE response."""
    event_name = "message"
    data_lines: list[str] = []

    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.rstrip("\n")
        if line == "":
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload == "[DONE]":
                    return
                try:
                    yield event_name, json.loads(payload)
                except json.JSONDecodeError:
                    yield event_name, payload
            event_name = "message"
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith("id:"):
            continue

    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            try:
                yield event_name, json.loads(payload)
            except json.JSONDecodeError:
                yield event_name, payload


def result_set_to_frame(result_set: dict[str, Any]):
    """Convert a Snowflake SQL API ResultSet dict into a pandas DataFrame."""
    import pandas as pd

    meta = (
        result_set.get("resultSetMetaData")
        or result_set.get("result_set_meta_data")
        or {}
    )
    row_type = meta.get("rowType") or meta.get("row_type") or []
    columns = [col.get("name", f"col_{i}") for i, col in enumerate(row_type)]
    data = result_set.get("data") or []
    if columns:
        return pd.DataFrame(data, columns=columns)
    return pd.DataFrame(data)
