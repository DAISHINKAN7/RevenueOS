"""Negotiation persistence (spec §85: "negotiation is included in audit history").

These tables hang off the *same* declarative `Base` as the recovery schema, so
`init_db()` creates them with no migration step and a negotiation that turns
into a checkout shares one database with the opportunity it may later become.

Every turn is written before the response is returned. There is no in-memory
negotiation state: if the process dies mid-negotiation the transcript survives,
and the reserve price is recomputed from product economics rather than recalled,
so a resumed negotiation cannot drift.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON, Boolean, DateTime, ForeignKey, Integer, Numeric, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models import Base, utcnow


class NegotiationSession(Base):
    """One buyer intent, from natural-language request to terminal state."""

    __tablename__ = "negotiation_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    # What the buyer agent said, and what was parsed out of it.
    buyer_request_text: Mapped[str] = mapped_column(Text, default="")
    constraints: Mapped[dict] = mapped_column(JSON, default=dict)
    constraints_source: Mapped[str] = mapped_column(String(16), default="RULE")  # LLM | RULE

    product_id: Mapped[str | None] = mapped_column(String(24), index=True)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    city_tier: Mapped[int] = mapped_column(Integer, default=1)

    status: Mapped[str] = mapped_column(String(32), default="OPEN", index=True)
    # OPEN | AGREED | ABANDONED | REJECTED | AWAITING_APPROVAL | CHECKED_OUT
    rounds: Mapped[int] = mapped_column(Integer, default=0)

    agreed_unit_price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    agreed_order_total: Mapped[float | None] = mapped_column(Numeric(12, 2))
    agreed_contribution: Mapped[float | None] = mapped_column(Numeric(12, 2))
    agreed_margin_percent: Mapped[float | None] = mapped_column(Numeric(6, 2))

    policy_version: Mapped[str] = mapped_column(String(32), default="")
    catalog_version: Mapped[str] = mapped_column(String(32), default="")

    # The bridge into Track 03. Null until checkout is created.
    opportunity_id: Mapped[str | None] = mapped_column(String(64), index=True)

    turns: Mapped[list["NegotiationTurn"]] = relationship(
        back_populates="session",
        order_by="NegotiationTurn.round_number",
        cascade="all, delete-orphan",
    )


class NegotiationTurn(Base):
    """One offer and one merchant ruling. Immutable once written."""

    __tablename__ = "negotiation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("negotiation_sessions.id", ondelete="CASCADE"), index=True)
    round_number: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    actor: Mapped[str] = mapped_column(String(16))  # BUYER | MERCHANT
    requested_unit_price: Mapped[float | None] = mapped_column(Numeric(12, 2))

    decision: Mapped[str | None] = mapped_column(String(24))
    reason_code: Mapped[str | None] = mapped_column(String(64))
    binding_constraint: Mapped[str | None] = mapped_column(String(64))
    offered_unit_price: Mapped[float | None] = mapped_column(Numeric(12, 2))
    margin_percent: Mapped[float | None] = mapped_column(Numeric(6, 2))
    net_contribution: Mapped[float | None] = mapped_column(Numeric(12, 2))

    # Full rule ledger, exactly as the engine produced it.
    rules: Mapped[list] = mapped_column(JSON, default=list)

    # Natural language. Explanatory only — never an input to pricing.
    message: Mapped[str] = mapped_column(Text, default="")
    message_source: Mapped[str] = mapped_column(String(16), default="TEMPLATE")  # LLM | TEMPLATE

    session: Mapped[NegotiationSession] = relationship(back_populates="turns")

    __table_args__ = (
        UniqueConstraint("session_id", "round_number", "actor", name="uq_negotiation_turn"),
    )