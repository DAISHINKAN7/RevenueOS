"""Hidden environmental mechanisms (spec Section 31).

These windows shift behaviour but are NEVER exposed as features. The model can
observe only their consequences (elevated failure rates, depressed conversion),
which is what stops it from recovering the generating process exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.config import SimulationConfig


@dataclass
class HiddenEnvironment:
    """Time-indexed hidden state, sampled once per dataset."""

    bank_outages: list[tuple[pd.Timestamp, pd.Timestamp]]
    competitor_sales: list[tuple[pd.Timestamp, pd.Timestamp]]
    courier_disruptions: list[tuple[pd.Timestamp, pd.Timestamp]]
    payday_days: tuple[int, ...]

    # ---- queries ---------------------------------------------------------
    def in_bank_outage(self, ts: pd.Series) -> np.ndarray:
        return self._in_windows(ts, self.bank_outages)

    def in_competitor_sale(self, ts: pd.Series) -> np.ndarray:
        return self._in_windows(ts, self.competitor_sales)

    def in_courier_disruption(self, ts: pd.Series) -> np.ndarray:
        return self._in_windows(ts, self.courier_disruptions)

    def is_payday(self, ts: pd.Series) -> np.ndarray:
        return np.isin(pd.DatetimeIndex(ts).day, self.payday_days)

    @staticmethod
    def _in_windows(ts: pd.Series, windows) -> np.ndarray:
        idx = pd.DatetimeIndex(ts)
        out = np.zeros(len(idx), dtype=bool)
        for start, end in windows:
            out |= (idx >= start) & (idx < end)
        return out

    def summary(self) -> dict:
        fmt = lambda ws: [(str(a), str(b)) for a, b in ws]  # noqa: E731
        return {
            "bank_outages": fmt(self.bank_outages),
            "competitor_sales": fmt(self.competitor_sales),
            "courier_disruptions": fmt(self.courier_disruptions),
            "payday_days": list(self.payday_days),
        }


def build_environment(cfg: SimulationConfig, rng: np.random.Generator) -> HiddenEnvironment:
    start = pd.Timestamp(cfg.start_date)
    horizon_hours = cfg.days * 24

    def windows(n: int, length_hours: int):
        out = []
        for _ in range(n):
            offset = int(rng.integers(0, max(1, horizon_hours - length_hours)))
            a = start + pd.Timedelta(hours=offset)
            out.append((a, a + pd.Timedelta(hours=length_hours)))
        return sorted(out)

    return HiddenEnvironment(
        bank_outages=windows(cfg.n_bank_outages, cfg.bank_outage_hours),
        competitor_sales=windows(cfg.n_competitor_sales, cfg.competitor_sale_days * 24),
        courier_disruptions=windows(cfg.n_courier_disruptions, cfg.courier_disruption_days * 24),
        payday_days=cfg.payday_days,
    )
