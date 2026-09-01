"""LLM backends for auto-interp explanation and scoring.

`OpenAICompatBackend` covers the OpenAI API and any OpenAI-compatible server
(vLLM `vllm serve`, in particular). `resolve_endpoint` implements the
endpoint-file discovery flow used with `scripts/slurm/launch_vllm.sbatch`:
the server job writes {host, port, model, job_id} to a JSON file; the client
polls that file, retries while the server is still loading, and fails loudly
if the slurm job has died.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests


class LLMBackend(Protocol):
    model: str

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str: ...


@dataclass
class OpenAICompatBackend:
    """Chat-completions client for OpenAI or any compatible endpoint."""

    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 5

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, ValueError) as error:
                last_error = error
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"LLM request failed after {self.max_retries} retries: {last_error}"
        )


@dataclass
class AnthropicBackend:
    """Anthropic Messages API client (secondary option)."""

    model: str
    timeout: float = 120.0
    max_retries: int = 5

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    json={
                        "model": self.model,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]
            except (requests.RequestException, KeyError, ValueError) as error:
                last_error = error
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"Anthropic request failed after {self.max_retries} retries: {last_error}"
        )


def _slurm_job_alive(job_id: str) -> bool:
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        state = out.stdout.strip()
        return state in ("RUNNING", "PENDING", "CONFIGURING")
    except (OSError, subprocess.TimeoutExpired):
        # squeue unavailable (e.g. off-cluster): assume alive, let HTTP decide.
        return True


def resolve_endpoint(
    endpoint_file: str | Path,
    wait_seconds: float = 1800.0,
    poll_seconds: float = 15.0,
) -> dict[str, str]:
    """Wait for the vLLM server's endpoint file and a healthy /v1/models.

    Returns the endpoint record {host, port, model, job_id}. Raises if the
    slurm job dies or the wait budget is exhausted.
    """
    endpoint_path = Path(endpoint_file)
    deadline = time.monotonic() + wait_seconds
    record: dict[str, str] | None = None
    while time.monotonic() < deadline:
        if endpoint_path.exists():
            try:
                record = json.loads(endpoint_path.read_text())
            except (json.JSONDecodeError, OSError):
                record = None
        if record:
            job_id = str(record.get("job_id", ""))
            if job_id and not _slurm_job_alive(job_id):
                raise RuntimeError(
                    f"vLLM server job {job_id} is no longer running "
                    f"(endpoint file {endpoint_path})"
                )
            base_url = f"http://{record['host']}:{record['port']}/v1"
            try:
                response = requests.get(f"{base_url}/models", timeout=10)
                if response.ok:
                    record["base_url"] = base_url
                    return record
            except requests.RequestException:
                pass  # still loading; keep polling
        time.sleep(poll_seconds)
    raise RuntimeError(
        f"vLLM endpoint at {endpoint_path} not healthy after {wait_seconds:.0f}s"
    )


def build_backend(
    backend: str,
    model: str,
    base_url: str = "",
    endpoint_file: str = "",
) -> LLMBackend:
    if backend == "anthropic":
        return AnthropicBackend(model=model)
    if backend != "openai_compat":
        raise ValueError(f"unknown backend {backend!r}")
    if endpoint_file:
        record = resolve_endpoint(endpoint_file)
        return OpenAICompatBackend(
            base_url=record["base_url"], model=model or record.get("model", "")
        )
    if not base_url:
        base_url = "https://api.openai.com/v1"
    return OpenAICompatBackend(base_url=base_url, model=model)
