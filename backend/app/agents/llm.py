"""LLM provider abstraction for the agent layer (spec §35-38).

Deliberately vendor-neutral and free-tier friendly. The HTTP provider speaks the
OpenAI-compatible chat-completions dialect, which is what Groq, OpenRouter,
Together and a local Ollama all expose — so switching providers is a base-URL
change, not a code change.

Defaults to `mock`, which is fully deterministic and used by CI. Nothing here is
required for correctness: with `AGENT_LLM_ENABLED=false` the deterministic
backend runs exactly as before.

Free options, for reference:
    Groq        https://api.groq.com/openai/v1     (free API key)
    OpenRouter  https://openrouter.ai/api/v1       (`:free` models)
    Ollama      http://localhost:11434/v1          (fully local, no key)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("revenueos.agent.llm")

AGENT_VERSION = "recovery-orchestrator-1.0.0"


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = os.getenv("AGENT_LLM_ENABLED", "false").lower() == "true"
    provider: str = os.getenv("AGENT_LLM_PROVIDER", "mock")
    base_url: str = os.getenv("AGENT_LLM_BASE_URL", "https://api.groq.com/openai/v1")
    api_key: str = os.getenv("AGENT_LLM_API_KEY", "")
    model: str = os.getenv("AGENT_LLM_MODEL", "llama-3.3-70b-versatile")
    timeout_seconds: float = float(os.getenv("AGENT_LLM_TIMEOUT", "20"))
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "12"))

    @property
    def active(self) -> bool:
        if not self.enabled or self.provider == "mock":
            return False
        # Ollama needs no key; hosted providers do.
        return bool(self.api_key) or "localhost" in self.base_url or "127.0.0.1" in self.base_url


class AgentLLMProvider(Protocol):
    name: str

    def decide_next_step(self, system_prompt: str, observation: dict) -> dict:
        """Return a structured next-step decision. Never free prose."""
        ...


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Small models wrap JSON in prose or fences despite instructions, so this is
    tolerant about framing but strict about the result: anything unparseable
    raises, and the caller falls back deterministically.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start:i + 1])
    raise ValueError("no JSON object in model response")


class MockLLMProvider:
    """Deterministic planner used by CI and by the no-key deployment path.

    Delegates to the same `fallback_decision` router the orchestrator uses when
    a real planner fails, so the two paths cannot diverge: a scenario that
    passes under the mock behaves identically under an LLM outage.

    A `script` may be supplied to replay fixed decisions in adversarial tests.
    """

    name = "mock"

    def __init__(self, script: list[dict] | None = None):
        self.script = list(script or [])
        self.calls = 0

    def decide_next_step(self, system_prompt: str, observation: dict) -> dict:
        self.calls += 1
        if self.script:
            return self.script.pop(0)
        from backend.app.agents.authorizer import fallback_decision

        decision = fallback_decision(
            observation.get("workflow_state", ""),
            set(observation.get("tools_called") or []))
        return decision.model_dump()


class OpenAICompatibleProvider:
    """Works with any OpenAI-compatible chat-completions endpoint."""

    name = "openai_compatible"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def _post(self, payload: dict, headers: dict):
        import httpx

        with httpx.Client(timeout=self.cfg.timeout_seconds) as client:
            return client.post(f"{self.cfg.base_url.rstrip('/')}/chat/completions",
                               json=payload, headers=headers)

    def decide_next_step(self, system_prompt: str, observation: dict) -> dict:
        # The instruction is repeated in the USER message, not left only in the
        # system message. Small local models attend far more strongly to the
        # last user turn, and when handed a bare JSON blob they tend to echo it
        # back rather than answer. Restating the required shape immediately
        # after the data measurably improves adherence on 3B-class models.
        tools = observation.get("available_tools") or []
        guidance = observation.get("state_guidance") or ""
        user_content = (
            "Here is the current workflow observation:\n\n"
            f"{json.dumps(observation, default=str)}\n\n"
            + (f"Guidance for this state: {guidance}\n\n" if guidance else "")
            + (f"`next_step` must be exactly one of: {', '.join(tools)}, "
               "STOP, WAIT, ESCALATE.\n\n" if tools else "")
            + "Do NOT repeat or echo the observation. Decide the single next step "
            "and reply with ONLY this JSON object:\n"
            '{"goal": "<short goal>", "observation_summary": "<one sentence>", '
            '"next_step": "<a tool name, or STOP or WAIT or ESCALATE>", '
            '"reason_code": "<SHORT_CODE>", '
            '"requires_financial_authorization": false}'
        )
        payload = {
            "model": self.cfg.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        r = self._post(payload, headers)

        # Older Ollama builds and some gateways reject `response_format`. Retry
        # once without it rather than falling back to the deterministic planner,
        # which would look like the LLM is working when it never ran.
        if r.status_code == 400:
            log.info("provider rejected response_format; retrying without it")
            payload.pop("response_format", None)
            r = self._post(payload, headers)

        if r.status_code >= 400:
            # The status alone is not diagnosable: 400 covers a bad model name,
            # 401 a bad key, 404 a wrong path, 429 a rate limit. Surface the
            # provider's own error message (which contains no credentials)
            # while never echoing the key or the full body.
            detail = ""
            try:
                body = r.json()
                err = body.get("error")
                if isinstance(err, dict):
                    detail = str(err.get("message") or err.get("type") or "")[:200]
                elif err:
                    detail = str(err)[:200]
            except Exception:  # noqa: BLE001
                detail = r.text[:200] if hasattr(r, "text") else ""
            raise RuntimeError(
                f"LLM provider returned HTTP {r.status_code}"
                + (f": {detail}" if detail else ""))

        body = r.json()
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError("unexpected provider response shape") from exc
        return _extract_json(content)

    def healthcheck(self) -> dict:
        """Verify the provider answers in the shape the orchestrator requires.

        Uses the real message construction, so a pass here means the planner
        path genuinely works rather than only that the server is reachable.
        """
        return self.decide_next_step(
            SYSTEM_PROMPT_PROBE,
            {"workflow_state": "RECOVERED", "tools_called": [],
             "available_tools": ["get_opportunity"],
             "note": "terminal state; the only valid next_step is STOP"})


SYSTEM_PROMPT_PROBE = (
    "You output JSON only. Never echo the input. The workflow state is "
    "terminal, so the correct next_step is exactly STOP.")


def get_provider(cfg: LLMConfig | None = None) -> AgentLLMProvider:
    cfg = cfg or LLMConfig()
    if not cfg.active:
        return MockLLMProvider()
    try:
        return OpenAICompatibleProvider(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("llm_provider_init_failed; falling back to mock: %s", type(exc).__name__)
        return MockLLMProvider()