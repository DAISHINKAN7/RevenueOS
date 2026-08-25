"""Verify the configured LLM provider actually responds.

    make agent-check

Fails loudly rather than silently degrading, so you never record a demo
believing the LLM ran when the deterministic planner was doing the work.
"""

from __future__ import annotations

import sys
import time

from backend.app.core.dotenv import load_dotenv

load_dotenv()  # before any settings are read

from backend.app.agents.llm import LLMConfig, MockLLMProvider, OpenAICompatibleProvider


def main() -> int:
    cfg = LLMConfig()
    print(f"AGENT_LLM_ENABLED  {cfg.enabled}")
    print(f"provider           {cfg.provider}")
    print(f"base_url           {cfg.base_url}")
    print(f"model              {cfg.model}")
    print(f"api key present    {bool(cfg.api_key)}")
    print(f"config active      {cfg.active}\n")

    if not cfg.active:
        print("Result: MOCK planner (deterministic).")
        print("The system is fully functional this way, but no LLM will be called.")
        print("For Ollama set: AGENT_LLM_ENABLED=true, AGENT_LLM_PROVIDER=ollama,")
        print("                AGENT_LLM_BASE_URL=http://localhost:11434/v1")
        return 0

    provider = OpenAICompatibleProvider(cfg)
    try:
        t = time.time()
        out = provider.healthcheck()
        elapsed = time.time() - t
    except Exception as exc:  # noqa: BLE001
        print(f"Result: FAILED — {type(exc).__name__}: {exc}")
        print("\nCommon causes:")
        print("  • ollama not running        -> run `ollama serve`")
        print("  • model not pulled          -> run `ollama pull <model>`")
        print("  • wrong base_url            -> must end in /v1")
        print("  • model name mismatch       -> check `ollama list`")
        print("\nThe agent will still work using the deterministic planner.")
        return 1

    print(f"Provider responded in {elapsed:.1f}s")
    print(f"parsed JSON: {out}\n")

    # Responding is not the same as being usable. The probe asks for one exact
    # object; a model that echoes the input or omits `next_step` will be
    # rejected by the orchestrator on every step and silently fall back to the
    # deterministic planner, which would make the LLM look active when it is not.
    required = {"next_step", "observation_summary", "reason_code"}
    missing = required - set(out)
    echoed = "workflow_state" in out or "available_tools" in out

    if missing or echoed or out.get("next_step") != "STOP":
        print("Result: RESPONDS BUT UNUSABLE")
        if echoed:
            print("  • the model echoed the input instead of answering the probe")
        if missing:
            print(f"  • missing required field(s): {sorted(missing)}")
        elif out.get("next_step") != "STOP":
            print(f"  • next_step was {out.get('next_step')!r}, expected 'STOP'")
        print("\nThe orchestrator will reject these responses and fall back to the")
        print("deterministic planner on every step, so the LLM would contribute")
        print("nothing. Use a larger instruct-tuned model:")
        print("  ollama pull qwen2.5:7b-instruct     # best JSON adherence, ~4.7GB")
        print("  ollama pull llama3.1:8b             # alternative, ~4.9GB")
        print("  ollama pull qwen2.5:3b-instruct     # if RAM is tight, ~2GB")
        print("\nThen set AGENT_LLM_MODEL to the exact name from `ollama list`.")
        return 1

    print("Result: OK — the model follows the required output contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())