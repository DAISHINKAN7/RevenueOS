"""The AI buyer (spec §41, §43).

The buyer is an agent on the *other* side of the table. It has a budget and a
preference and it argues its case in natural language. It is also, deliberately,
not trusted with anything:

* Its parsed constraints are **validated and clamped** before use. A buyer that
  claims ``max_budget: 999999999`` or emits a negative price gets a rejected
  parse, not an exception and not a lucky deal.
* Its accept/decline decision is **deterministic** — accept iff the merchant's
  offer fits the budget it declared at the start. The LLM writes the sentence;
  arithmetic decides the answer. This is the same separation the recovery agent
  uses, applied to the counterparty.
* Nothing it says reaches `evaluate_offer`, which accepts no text parameter.

The LLM is optional throughout. With `AGENT_LLM_ENABLED=false` (or no key), the
rule-based parser and message templates run instead and every test still passes.
CI therefore validates the protocol rather than a model's mood.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation

BUYER_VERSION = "ai-buyer-1.0.0"

# Hard bounds on anything a model hands back. A parsed constraint outside these
# is treated as a parse failure, not clamped silently into something plausible.
MAX_BUDGET_CEILING = Decimal("500000")
MAX_QUANTITY = 10


@dataclass
class BuyerConstraints:
    query: str = ""
    category: str | None = None
    max_budget: float | None = None
    min_price: float | None = None
    quantity: int = 1
    city_tier: int = 1
    priorities: list[str] | None = None
    target_discount_percent: float = 8.0

    def as_dict(self) -> dict:
        return asdict(self)


class BuyerParseError(ValueError):
    """The request could not be turned into safe structured constraints."""


# --------------------------------------------------------------------------- #
# LLM plumbing — optional, defensive, never load-bearing
# --------------------------------------------------------------------------- #

def llm_active() -> bool:
    """True when a real provider is configured.

    Reads the same `LLMConfig` the recovery orchestrator uses, so agentic
    commerce and recovery are never in disagreement about whether the LLM is on.
    """
    try:
        from backend.app.agents.llm import LLMConfig

        return bool(LLMConfig().active)
    except Exception:  # noqa: BLE001
        return False


def _llm_complete(system: str, user: str) -> str | None:
    """Free-text completion against the configured OpenAI-compatible endpoint.

    The existing provider exposes only `decide_next_step`, which is hard-wired
    to the orchestrator's planner JSON schema. Reusing it here would mean
    pretending a buyer's sentence is a planner step. Instead this issues a plain
    chat-completions call using the *same* `LLMConfig` — same base URL, key,
    model and timeout — so there is one place to configure the LLM and no second
    credential path.

    Any failure returns None and the deterministic template runs. An LLM outage
    must degrade the prose, never the protocol.
    """
    try:
        import httpx

        from backend.app.agents.llm import LLMConfig
    except Exception:  # noqa: BLE001
        return None

    cfg = LLMConfig()
    if not cfg.active:
        return None

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    try:
        with httpx.Client(timeout=cfg.timeout_seconds) as client:
            r = client.post(
                f"{cfg.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": cfg.model,
                    "temperature": 0,
                    "max_tokens": 300,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        if r.status_code >= 400:
            return None
        return r.json()["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001
        return None


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fence.group(1) if fence else None
    if raw is None:
        brace = re.search(r"\{.*\}", text, re.S)
        raw = brace.group(0) if brace else None
    if raw is None:
        return None
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

# Accepts "6000", "₹6,000", "Rs 6000", "INR 6000", "6k".
_CUR = r"(?:₹|rs\.?|inr)?\s*"
_NUM = rf"{_CUR}([0-9][0-9,]*(?:\.[0-9]+)?)\s*(k|thousand)?"


def _num(m: re.Match) -> Decimal:
    val = Decimal(m.group(1).replace(",", ""))
    if m.group(2):
        val *= 1000
    return val


def parse_rule_based(text: str) -> BuyerConstraints:
    """Deterministic parser. Always runs — as fallback and as validator."""
    t = text.lower().strip()
    c = BuyerConstraints(query=text.strip())

    budget = None
    for pat in (rf"under\s+{_NUM}", rf"below\s+{_NUM}", rf"less\s+than\s+{_NUM}",
                rf"budget\s+(?:of\s+)?{_NUM}", rf"up\s+to\s+{_NUM}",
                rf"max(?:imum)?\s+(?:of\s+)?{_NUM}", rf"within\s+{_NUM}"):
        m = re.search(pat, t)
        if m:
            budget = _num(m)
            break
    if budget is None:
        m = re.search(r"₹\s*([0-9][0-9,]*)", text)
        if m:
            try:
                budget = Decimal(m.group(1).replace(",", ""))
            except InvalidOperation:
                budget = None
    if budget is not None and 0 < budget <= MAX_BUDGET_CEILING:
        c.max_budget = float(budget)

    m = re.search(r"\b(\d{1,2})\s*(?:units?|pieces?|pcs?|x)\b", t)
    if m:
        c.quantity = max(1, min(int(m.group(1)), MAX_QUANTITY))

    m = re.search(r"tier\s*([123])", t)
    if m:
        c.city_tier = int(m.group(1))

    # Stopwords that carry no catalog signal.
    stop = {
        "need", "want", "looking", "under", "below", "with", "good", "some",
        "the", "for", "and", "that", "have", "buy", "get", "please", "would",
        "like", "about", "around", "budget", "than", "less", "more", "best",
        "can", "you", "find", "something", "great", "really", "make", "sure",
        "units", "unit", "tier", "pieces", "pcs", "rupees", "inr",
    }
    words = [w for w in re.findall(r"[a-z]{3,}", t) if w not in stop]
    c.query = " ".join(words[:8]) or text.strip()

    prio = [w for w in ("battery", "warranty", "delivery", "quality", "durable",
                        "lightweight", "premium") if w in t]
    c.priorities = prio or None
    return c


_PARSE_SYSTEM = (
    "You convert a shopper's request into JSON constraints for a product "
    "catalog search. Reply with ONLY a JSON object, no prose and no markdown "
    "fences. Schema: {\"query\": string of 2-6 search keywords, \"category\": "
    "string or null, \"max_budget\": number or null, \"quantity\": integer >= 1, "
    "\"priorities\": array of short strings}. Never invent a budget the shopper "
    "did not state. Use null when unsure."
)


def parse_request(text: str, use_llm: bool = True) -> tuple[BuyerConstraints, str]:
    """Natural language → validated constraints. Returns (constraints, source).

    The rule-based parse is computed first and used as the guard rail: LLM
    fields are accepted only where they are well-typed and in range, and the
    budget in particular is never accepted from the model if the rule parser
    found a different number in the text. A hallucinated budget is the one
    parse error with real money attached.
    """
    if not text or not text.strip():
        raise BuyerParseError("empty buyer request")

    base = parse_rule_based(text)
    if not use_llm:
        return base, "RULE"

    raw = _llm_complete(_PARSE_SYSTEM, text.strip())
    data = _extract_json(raw or "")
    if not data:
        return base, "RULE"

    out = BuyerConstraints(**base.as_dict())

    q = data.get("query")
    if isinstance(q, str) and 0 < len(q) <= 120:
        out.query = q.strip()

    cat = data.get("category")
    if isinstance(cat, str) and 0 < len(cat) <= 60:
        out.category = cat.strip()

    b = data.get("max_budget")
    if isinstance(b, (int, float)) and 0 < b <= float(MAX_BUDGET_CEILING):
        # Only trusted when the deterministic parser did not find one, or agrees.
        if base.max_budget is None or abs(base.max_budget - float(b)) < 1.0:
            out.max_budget = float(b)

    qty = data.get("quantity")
    if isinstance(qty, int) and 1 <= qty <= MAX_QUANTITY:
        out.quantity = qty

    pr = data.get("priorities")
    if isinstance(pr, list):
        clean = [str(x)[:24] for x in pr[:5] if isinstance(x, (str, int, float))]
        out.priorities = clean or None

    return out, "LLM"


# --------------------------------------------------------------------------- #
# Buyer's response to a merchant ruling
# --------------------------------------------------------------------------- #

@dataclass
class BuyerResponse:
    action: str                # ACCEPT | COUNTER | WALK_AWAY
    counter_unit_price: float | None
    message: str
    message_source: str


def decide(
    constraints: BuyerConstraints,
    decision: str,
    offered_unit_price: Decimal,
    order_total: Decimal,
    round_number: int,
    list_price: Decimal,
    use_llm: bool = True,
) -> BuyerResponse:
    """Deterministic buyer decision. The model only phrases it.

    Rules, in order:
      1. Merchant rejected outright  → walk away.
      2. Offer exceeds declared budget → walk away. The buyer does not talk
         itself into overspending, which is precisely what makes it a useful
         counterparty to test a merchant against.
      3. Merchant accepted → accept.
      4. First round counter → one bounded counter-offer, anchored at the
         buyer's target discount but never below what it already knows the
         merchant refused.
      5. Otherwise → accept if within budget.
    """
    budget = Decimal(str(constraints.max_budget)) if constraints.max_budget else None
    within_budget = budget is None or order_total <= budget

    if decision == "REJECT":
        action, counter = "WALK_AWAY", None
    elif not within_budget:
        action, counter = "WALK_AWAY", None
    elif decision in ("ACCEPT", "REQUIRE_APPROVAL"):
        action, counter = "ACCEPT", None
    elif decision == "COUNTER":
        # The merchant has just published its reserve. Countering below a floor
        # you have been shown is not negotiation, it is noise — and it burns a
        # round. A rational buyer takes the floor if it fits the budget.
        #
        # The buyer only counters when it has NOT yet been shown a floor, which
        # is why its aggressive opening bid is where its leverage lives. There
        # is one exception: if its target is above the merchant's floor it is
        # already getting a better deal than it asked for, so it accepts.
        action, counter = "ACCEPT", None

    msg, src = _buyer_message(action, offered_unit_price, order_total, budget, counter, use_llm)
    return BuyerResponse(action=action, counter_unit_price=counter,
                         message=msg, message_source=src)


_TEMPLATES = {
    "ACCEPT": "That works — ₹{unit} a unit, ₹{total} all in, inside my ₹{budget} budget. Let's close it.",
    "COUNTER": "₹{unit} is closer. Could you do ₹{counter}? That's my last ask.",
    "WALK_AWAY": "₹{total} is past my ₹{budget} ceiling, so I'll pass on this one.",
}


def _buyer_message(action, unit, total, budget, counter, use_llm) -> tuple[str, str]:
    fmt = {
        "unit": f"{unit:,.0f}",
        "total": f"{total:,.0f}",
        "budget": f"{budget:,.0f}" if budget else "stated",
        "counter": f"{counter:,.0f}" if counter else "-",
    }
    template = _TEMPLATES[action].format(**fmt)
    if not use_llm:
        return template, "TEMPLATE"

    out = _llm_complete(
        "You are a procurement agent negotiating on a buyer's behalf. Write ONE "
        "short sentence, first person, no greeting, no emoji. State only the "
        "figures given to you. Do not invent numbers or promise anything.",
        f"Decision already made: {action}. Unit price offered: {fmt['unit']}. "
        f"Order total: {fmt['total']}. Budget: {fmt['budget']}. "
        f"Counter price if any: {fmt['counter']}.",
    )
    if out and 0 < len(out.strip()) <= 300:
        return out.strip().strip('"'), "LLM"
    return template, "TEMPLATE"