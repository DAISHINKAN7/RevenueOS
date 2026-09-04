# Agentic Commerce — Track 01

Closes spec §41–§43 and the §85 Definition of Done.

## The claim

The merchant economics and policy layer built for revenue recovery can safely
negotiate with AI buyers. Not "we bolted a chatbot onto a catalog" — the same
contribution-margin arithmetic that decides whether a discount is worth offering
decides what price a buyer agent may be quoted.

## What is deterministic and what is not

| Concern | Owner | Can an LLM change it? |
|---|---|---|
| Reserve price | `services/negotiation.py` | No |
| ACCEPT / COUNTER / REJECT | `services/negotiation.py` | No |
| Margin, contribution, discount | `services/negotiation.py` | No |
| Buyer accept / walk away | `agents/buyer.py` (arithmetic) | No |
| Parsing "under ₹6,000" | `agents/buyer.py` | Yes, but validated and clamped |
| The sentences on screen | both | Yes |

`evaluate_offer()` **accepts no text parameter**. A test asserts the signature,
so a future refactor cannot quietly open a channel from buyer prose into
merchant pricing. This is the same structural defence `PolicyEngine.evaluate()`
uses on the recovery side.

## The reserve price

For unit price `R`, quantity `q`, shipping fee `fee`, fulfilment cost `F`,
return rate `r`, COGS `C`:

```
gross_contribution = R·q + fee − C·q − F
net_contribution   = (1−r)·gross_contribution − r·F
margin%            = 100 · net_contribution / (R·q + fee)
```

A return costs the merchant the contribution it booked *and* reverse logistics —
hence `− r·F`.

Three price floors are computed, and the reserve is their maximum:

| Rule | Floor |
|---|---|
| `RULE_AC_DISCOUNT_PERCENT_CEILING` | `list × (1 − max_autonomous_discount_percent)` |
| `RULE_AC_DISCOUNT_AMOUNT_CEILING` | `list − max_autonomous_discount_amount/q` |
| `RULE_AC_MARGIN_FLOOR` | closed-form solve of `margin% ≥ minimum_remaining_contribution_margin_percent` |

The margin floor solves to:

```
R ≥ ( [ (1−r)(Cq + F) + rF ] / (1 − r − m)  −  fee ) / q     where m = min_margin/100
```

If `1 − r − m ≤ 0` the floor is unreachable at any price and the SKU is rejected.

**Two reserves, not one.** `reserve_hard` is the lowest lawful price.
`reserve_autonomous` is raised further so the discount also stays under
`human_approval_required_above_discount` — that is what gets quoted, so a
counter the buyer accepts is always immediately executable. A request landing
*between* the two is a genuine `REQUIRE_APPROVAL`: lawful, but not the agent's
call.

## Decision table

| Condition | Decision |
|---|---|
| requested ≥ list | `ACCEPT` at list — overpayment is a refund liability, not upside |
| requested ≥ reserve_autonomous | `ACCEPT` |
| reserve_hard ≤ requested < reserve_autonomous | `REQUIRE_APPROVAL` |
| requested < reserve_hard | `COUNTER` at reserve_autonomous |
| reserve > list price | `REJECT` — no lawful price exists for this SKU |
| lowest lawful price > buyer's budget | `REJECT` — no zone of agreement |
| quantity > inventory | `REJECT` |
| round > 3 | `REJECT` |
| order total > high_value_order_threshold | `REQUIRE_APPROVAL` |

## The bridge to Track 03

`POST /api/agent-commerce/checkout` is deliberately thin. It does not
reimplement payments. It creates a normal `Opportunity` in `DETECTED` with a
context built from the negotiated economics and returns its id.

From that point the negotiated cart is indistinguishable from any other
opportunity — same predictor, same policy engine, same Razorpay Test Mode path.
If the payment fails, the same recovery loop takes over, now constrained by the
margin the negotiation already consumed. That is the §43 narrative in one
continuous flow.

## What this is not

- Not free-form haggling. Three rounds, hard-capped.
- Not a price the model chose. Ask twice, get the same number.
- Not a full agent-to-agent protocol. It is a REST offer protocol; §86 stretch
  goal 8 would extend it.
- The buyer agent is our own. It is a useful adversary, not an independent
  party, and the evaluation claims nothing about real buyer behaviour.

## Honest limitations

- Shipping fee policy (free over ₹999, else ₹85) is a fixed merchant term, not
  learned and not negotiable.
- The buyer's willingness to pay is rule-based. There is no model of buyer
  utility, so "the buyer accepted" is not evidence the price was optimal — only
  that it was lawful and within a declared budget.
- No inventory reservation. Two concurrent negotiations for the last unit can
  both reach `AGREED`. Correct for a demo, wrong for production.
- Negotiated opportunities carry a no-history customer context, because a
  negotiated cart genuinely has no purchase history. The predictor's estimates
  there are wider than for seeded opportunities.