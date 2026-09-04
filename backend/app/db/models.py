"""Persistence layer.

Design notes that matter for correctness:

* `RecoveryExecution.idempotency_key` carries a UNIQUE constraint. Duplicate
  protection lives in the database, not in application memory, because two
  concurrent workers do not share memory.
* `AuditEvent` is append-only by convention and by API: corrections are new
  AUDIT_CORRECTION rows, never updates.
* `WebhookInbox` has UNIQUE(provider, event_id) so a replayed Razorpay delivery
  cannot mutate financial state twice.
* Money is stored as Numeric, not float.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Index, Integer, JSON, Numeric, String,
    Text, UniqueConstraint, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from backend.app.core.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


MONEY = Numeric(14, 2)


class Opportunity(Base):
    __tablename__ = "opportunities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_checkout_id: Mapped[str | None] = mapped_column(String(64))
    customer_id: Mapped[str | None] = mapped_column(String(64))
    opportunity_type: Mapped[str] = mapped_column(String(32), index=True)

    detected_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    state: Mapped[str] = mapped_column(String(40), index=True)
    workflow_version: Mapped[str] = mapped_column(String(40))
    execution_mode: Mapped[str] = mapped_column(String(20), default="SIMULATOR")

    revenue_at_risk: Mapped[Decimal] = mapped_column(MONEY)
    contribution_margin_at_risk: Mapped[Decimal] = mapped_column(MONEY)

    current_attempt: Mapped[int] = mapped_column(Integer, default=1)
    selected_action: Mapped[str | None] = mapped_column(String(40))
    trace_id: Mapped[str] = mapped_column(String(64), index=True)

    # Free-text customer-supplied field. Deliberately present so adversarial
    # tests can prove it has zero influence on authorization.
    customer_note: Mapped[str | None] = mapped_column(Text)

    context: Mapped[dict] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    version_number: Mapped[int] = mapped_column(Integer, default=0)

    __mapper_args__ = {"version_id_col": version_number}  # optimistic locking

    executions: Mapped[list["RecoveryExecution"]] = relationship(back_populates="opportunity")


class ActionPrediction(Base):
    __tablename__ = "action_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[str] = mapped_column(ForeignKey("opportunities.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    action: Mapped[str] = mapped_column(String(40))
    probability: Mapped[float | None] = mapped_column(Numeric(8, 6))
    valid: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(String(120))

    model_version: Mapped[str] = mapped_column(String(64))
    feature_pipeline_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ActionFinancialEvaluation(Base):
    __tablename__ = "action_financial_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[str] = mapped_column(ForeignKey("opportunities.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    action: Mapped[str] = mapped_column(String(40))
    recovery_probability: Mapped[Decimal] = mapped_column(Numeric(8, 6))

    base_contribution_margin: Mapped[Decimal] = mapped_column(MONEY)
    incentive_cost_if_recovered: Mapped[Decimal] = mapped_column(MONEY)
    fixed_action_cost: Mapped[Decimal] = mapped_column(MONEY)
    expected_return_loss: Mapped[Decimal] = mapped_column(MONEY)
    expected_cancellation_loss: Mapped[Decimal] = mapped_column(MONEY)

    expected_value: Mapped[Decimal] = mapped_column(MONEY)
    incremental_expected_value: Mapped[Decimal] = mapped_column(MONEY)
    rank: Mapped[int | None] = mapped_column(Integer)

    calculation_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PolicyEvaluation(Base):
    __tablename__ = "policy_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[str] = mapped_column(ForeignKey("opportunities.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)

    action: Mapped[str] = mapped_column(String(40))
    policy_version: Mapped[str] = mapped_column(String(40))
    decision: Mapped[str] = mapped_column(String(24))
    maximum_authorized_downside: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    reason_code: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    rules: Mapped[list["PolicyRuleEvaluation"]] = relationship(back_populates="evaluation")


class PolicyRuleEvaluation(Base):
    __tablename__ = "policy_rule_evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_evaluation_id: Mapped[int] = mapped_column(ForeignKey("policy_evaluations.id"), index=True)

    rule_id: Mapped[str] = mapped_column(String(64))
    passed: Mapped[bool] = mapped_column(Boolean)
    decision: Mapped[str] = mapped_column(String(24))
    input_value: Mapped[str | None] = mapped_column(String(64))
    threshold: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)

    evaluation: Mapped[PolicyEvaluation] = relationship(back_populates="rules")


class RecoveryExecution(Base):
    __tablename__ = "recovery_executions"

    execution_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    opportunity_id: Mapped[str] = mapped_column(ForeignKey("opportunities.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)

    action: Mapped[str] = mapped_column(String(40))
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    execution_provider: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(32), index=True)

    amount: Mapped[Decimal | None] = mapped_column(MONEY)
    currency: Mapped[str] = mapped_column(String(8), default="INR")

    external_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    external_payment_id: Mapped[str | None] = mapped_column(String(64), index=True)
    external_payment_link_id: Mapped[str | None] = mapped_column(String(64))

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    opportunity: Mapped[Opportunity] = relationship(back_populates="executions")

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_execution_idempotency"),
        Index("ix_exec_opp_attempt", "opportunity_id", "attempt_number"),
    )


class RecoveryOutcome(Base):
    """One row per recovered opportunity. UNIQUE prevents double-counting."""

    __tablename__ = "recovery_outcomes"

    opportunity_id: Mapped[str] = mapped_column(
        ForeignKey("opportunities.id"), primary_key=True)
    execution_id: Mapped[str] = mapped_column(String(64))

    gross_order_value: Mapped[Decimal] = mapped_column(MONEY)
    discount_amount: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    shipping_subsidy: Mapped[Decimal] = mapped_column(MONEY, default=Decimal("0"))
    net_recovered_gmv: Mapped[Decimal] = mapped_column(MONEY)
    realized_contribution: Mapped[Decimal] = mapped_column(MONEY)

    payment_id: Mapped[str | None] = mapped_column(String(64))
    order_id: Mapped[str | None] = mapped_column(String(64))
    recovery_timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PaymentFailureRecord(Base):
    __tablename__ = "payment_failures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[str] = mapped_column(ForeignKey("opportunities.id"), index=True)
    execution_id: Mapped[str | None] = mapped_column(String(64))

    failure_code: Mapped[str | None] = mapped_column(String(64))
    failure_description: Mapped[str | None] = mapped_column(Text)
    failure_source: Mapped[str | None] = mapped_column(String(40))
    failure_step: Mapped[str | None] = mapped_column(String(40))
    payment_method: Mapped[str | None] = mapped_column(String(32))
    payment_id: Mapped[str | None] = mapped_column(String(64))
    provider_timestamp: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditEvent(Base):
    """Append-only. Never UPDATE a row here."""

    __tablename__ = "audit_events"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), index=True)
    opportunity_id: Mapped[str] = mapped_column(String(64), index=True)
    sequence_number: Mapped[int] = mapped_column(Integer)

    event_type: Mapped[str] = mapped_column(String(48), index=True)
    actor_type: Mapped[str] = mapped_column(String(24))
    actor_id: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    workflow_state_before: Mapped[str | None] = mapped_column(String(40))
    workflow_state_after: Mapped[str | None] = mapped_column(String(40))

    summary: Mapped[str] = mapped_column(Text)
    structured_payload: Mapped[dict] = mapped_column(JSON, default=dict)

    model_version: Mapped[str | None] = mapped_column(String(64))
    policy_version: Mapped[str | None] = mapped_column(String(40))
    execution_id: Mapped[str | None] = mapped_column(String(64))

    __table_args__ = (
        UniqueConstraint("opportunity_id", "sequence_number", name="uq_audit_sequence"),
    )


class WebhookInbox(Base):
    __tablename__ = "webhook_inbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(24))
    event_id: Mapped[str] = mapped_column(String(128), index=True)
    event_type: Mapped[str] = mapped_column(String(48))

    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    payload_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    signature_valid: Mapped[bool] = mapped_column(Boolean, default=False)

    processing_status: Mapped[str] = mapped_column(String(20), default="RECEIVED", index=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime)
    processing_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)

    provider_created_at: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        UniqueConstraint("provider", "event_id", name="uq_webhook_event"),
    )


SCHEMA_VERSION = "backend-schema-1.0.0"

_engine = None
_Session = None


def get_engine():
    global _engine
    if _engine is None:
        url = settings.database_url
        kwargs = {"future": True}
        if url.startswith("sqlite"):
            from pathlib import Path
            Path("data").mkdir(exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragma(conn, _):
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
    return _engine


def get_session_factory():
    global _Session
    if _Session is None:
        _Session = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _Session


def reset_sqlite_files(url: str | None = None) -> None:
    """Delete the SQLite database and its WAL sidecars together.

    WAL mode writes `-wal` and `-shm` alongside the main file. Removing only the
    main file leaves those orphaned, and the next connection fails with an
    opaque `disk I/O error`. They must be removed as a set.
    """
    from pathlib import Path

    url = url or settings.database_url
    if not url.startswith("sqlite"):
        return
    main = url.split("///")[-1]
    if not main or main == ":memory:":
        return
    for suffix in ("", "-wal", "-shm", "-journal"):
        f = Path(main + suffix)
        if f.exists():
            f.unlink()


# Importing the negotiation tables here registers them on Base.metadata
# so init_db() creates them with no migration step.
def _register_commerce_models() -> None:
    from backend.app.db import commerce_models  # noqa: F401


def init_db(drop: bool = False) -> None:
    _register_commerce_models()
    eng = get_engine()
    if drop:
        try:
            Base.metadata.drop_all(eng)
        except Exception:  # noqa: BLE001
            # A corrupted or half-deleted SQLite file cannot be introspected.
            # Rebuilding from scratch is correct here: this database holds
            # operational demo state only, never research artifacts.
            eng.dispose()
            reset_engine()
            reset_sqlite_files()
            eng = get_engine()
    Base.metadata.create_all(eng)


def reset_engine() -> None:
    """Used by tests that point DATABASE_URL somewhere else."""
    global _engine, _Session
    _engine = None
    _Session = None


class AgentRun(Base):
    """One orchestrator run over one opportunity (spec §25)."""

    __tablename__ = "agent_runs"

    agent_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    opportunity_id: Mapped[str] = mapped_column(ForeignKey("opportunities.id"), index=True)
    agent_version: Mapped[str] = mapped_column(String(40))

    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)

    status: Mapped[str] = mapped_column(String(24), default="RUNNING", index=True)
    current_goal: Mapped[str | None] = mapped_column(String(64))
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    final_disposition: Mapped[str | None] = mapped_column(String(40))

    planner_source: Mapped[str] = mapped_column(String(16), default="FALLBACK")
    initial_state: Mapped[str | None] = mapped_column(String(40))
    final_state: Mapped[str | None] = mapped_column(String(40))
    blocked_tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    planner_failures: Mapped[int] = mapped_column(Integer, default=0)
    budget_exceeded: Mapped[bool] = mapped_column(Boolean, default=False)

    llm_provider: Mapped[str] = mapped_column(String(32), default="mock")
    llm_model: Mapped[str | None] = mapped_column(String(64))
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    replan_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class AgentTraceEvent(Base):
    """Concise, user-safe decision summaries. Never hidden chain-of-thought."""

    __tablename__ = "agent_trace_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.agent_run_id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    event_type: Mapped[str] = mapped_column(String(40))
    tool_name: Mapped[str | None] = mapped_column(String(48))
    tool_input_summary: Mapped[str | None] = mapped_column(Text)
    tool_output_summary: Mapped[str | None] = mapped_column(Text)
    reasoning_summary: Mapped[str | None] = mapped_column(Text)

    workflow_state: Mapped[str | None] = mapped_column(String(40))
    policy_state: Mapped[str | None] = mapped_column(String(24))
