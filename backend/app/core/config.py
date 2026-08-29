"""Backend configuration and versioned merchant policy.

Every version string that can affect a financial decision lives here, so an
audit row can be traced back to the exact code and policy that produced it.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field

APPLICATION_VERSION = "0.5.0"
WORKFLOW_VERSION = "recovery-workflow-1.0.0"
FINANCIAL_ENGINE_VERSION = "financial-engine-1.0.0"
POLICY_VERSION = "merchant-policy-1.0.0"

ART = Path("ml/artifacts")
POLICY_FILE = Path("backend/app/core/merchant_policy.json")


class MerchantPolicy(BaseModel):
    """Deterministic financial authority granted to the agent.

    Nothing in this object is inferred by a model. It is merchant-configured
    data, versioned so that any past decision can be replayed against the
    policy that was actually in force.
    """

    policy_version: str = POLICY_VERSION

    max_autonomous_discount_percent: Decimal = Decimal("7")
    max_autonomous_discount_amount: Decimal = Decimal("300")
    max_free_shipping_cost: Decimal = Decimal("150")

    minimum_remaining_contribution_margin_percent: Decimal = Decimal("15")
    minimum_incremental_expected_value: Decimal = Decimal("0")
    minimum_recovery_probability: float = 0.20
    minimum_decision_margin: Decimal = Decimal("20")

    max_recovery_attempts: int = 2
    opportunity_ttl_minutes: int = 2880

    high_value_order_threshold: Decimal = Decimal("10000")
    human_approval_required_above_discount_amount: Decimal = Decimal("250")
    human_approval_required_for_high_value_orders: bool = True

    @classmethod
    def load(cls) -> "MerchantPolicy":
        if POLICY_FILE.exists():
            return cls(**json.loads(POLICY_FILE.read_text()))
        return cls()


class Settings(BaseModel):
    database_url: str = Field(default_factory=lambda: os.getenv(
        "DATABASE_URL", "sqlite:///data/revenueos.db"))
    admin_token: str = Field(default_factory=lambda: os.getenv("ADMIN_TOKEN", "dev-admin-token"))
    frontend_url: str = Field(default_factory=lambda: os.getenv("FRONTEND_URL", "http://localhost:3000"))

    # Razorpay — TEST MODE ONLY. Never populate with live keys.
    razorpay_key_id: str = Field(default_factory=lambda: os.getenv("RAZORPAY_KEY_ID", ""))
    razorpay_key_secret: str = Field(default_factory=lambda: os.getenv("RAZORPAY_KEY_SECRET", ""))
    razorpay_webhook_secret: str = Field(default_factory=lambda: os.getenv("RAZORPAY_WEBHOOK_SECRET", ""))
    razorpay_mode: str = Field(default_factory=lambda: os.getenv("RAZORPAY_MODE", "test"))
    razorpay_client: str = Field(default_factory=lambda: os.getenv("RAZORPAY_CLIENT", "mock"))
    razorpay_timeout_seconds: float = 10.0

    # Presentation pacing for the live SSE stream, in milliseconds between
    # stages. The work is real and already complete when each event fires; this
    # only spaces the events out so a viewer can follow them. Zero in tests and
    # CI, so it never affects correctness or timing-sensitive assertions.
    stream_pacing_ms: int = Field(
        default_factory=lambda: int(os.getenv("STREAM_PACING_MS", "0")))

    merchant_display_name: str = "NovaCart"
    currency: str = "INR"

    @property
    def razorpay_configured(self) -> bool:
        return bool(self.razorpay_key_id and self.razorpay_key_secret)

    def safe_dict(self) -> dict:
        """Never expose secrets, not even truncated."""
        return {
            "razorpay_mode": self.razorpay_mode,
            "razorpay_client": self.razorpay_client,
            "razorpay_configured": self.razorpay_configured,
            "payment_environment": "TEST",
            "currency": self.currency,
        }


settings = Settings()