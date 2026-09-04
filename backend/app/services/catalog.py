"""Machine-readable product catalog for agentic commerce (spec §41).

Reads the *same* 60 SKUs the simulator generated — `data/generated/products.parquet`
— so catalog economics and recovery economics are the same numbers. There is no
second product source and no hand-written demo catalog.

What an AI buyer is allowed to see and what the merchant reasons over are
deliberately different objects:

    PublicProduct    price, name, category, weight, stock, rating-ish signals
    ProductEconomics price, COGS, fulfilment cost, return rate

`ProductEconomics` never crosses the API boundary. A buyer agent that could read
COGS could compute the merchant's reserve price exactly, and the negotiation
would be theatre.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd

PRODUCTS_PARQUET = Path("data/generated/products.parquet")

# Zone multipliers by city tier — identical to ml/simulation/products.py.
ZONE_MULTIPLIER = {1: Decimal("1.00"), 2: Decimal("1.25"), 3: Decimal("1.60")}

CATALOG_VERSION = "catalog-1.0.0"


class CatalogUnavailable(RuntimeError):
    """Raised when the generated dataset is missing. Never silently faked."""


@dataclass(frozen=True)
class ProductEconomics:
    """Merchant-private economics. MUST NOT be serialized to a buyer."""

    product_id: str
    selling_price: Decimal
    cost_of_goods: Decimal
    base_shipping_cost: Decimal
    historical_return_rate: Decimal
    inventory_level: int

    def fulfilment_cost(self, city_tier: int = 1) -> Decimal:
        mult = ZONE_MULTIPLIER.get(int(city_tier), Decimal("1.00"))
        return (self.base_shipping_cost * mult).quantize(Decimal("0.01"))


@dataclass(frozen=True)
class PublicProduct:
    """The buyer-visible projection."""

    product_id: str
    name: str
    category: str
    subcategory: str
    list_price: float
    weight_kg: float
    in_stock: bool
    inventory_level: int
    bundle_group: str

    def as_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "category": self.category,
            "subcategory": self.subcategory,
            "list_price": self.list_price,
            "currency": "INR",
            "weight_kg": self.weight_kg,
            "in_stock": self.in_stock,
            "inventory_level": self.inventory_level,
            "bundle_group": self.bundle_group,
        }


@functools.lru_cache(maxsize=1)
def _frame() -> pd.DataFrame:
    if not PRODUCTS_PARQUET.exists():
        raise CatalogUnavailable(
            f"{PRODUCTS_PARQUET} not found. Run `make data` first — the catalog "
            "is the generated product set, not a separate fixture."
        )
    df = pd.read_parquet(PRODUCTS_PARQUET)
    required = {
        "product_id", "name", "category", "subcategory", "selling_price",
        "cost_of_goods", "base_shipping_cost", "historical_return_rate",
        "inventory_level", "weight_kg", "bundle_group",
    }
    missing = required - set(df.columns)
    if missing:
        raise CatalogUnavailable(f"products.parquet is missing columns: {sorted(missing)}")
    return df


def reset_cache() -> None:
    """Tests point the loader at a different dataset."""
    _frame.cache_clear()


def _to_public(row) -> PublicProduct:
    return PublicProduct(
        product_id=str(row.product_id),
        name=str(row.name),
        category=str(row.category),
        subcategory=str(row.subcategory),
        list_price=float(row.selling_price),
        weight_kg=float(row.weight_kg),
        in_stock=int(row.inventory_level) > 0,
        inventory_level=int(row.inventory_level),
        bundle_group=str(row.bundle_group),
    )


def list_products(limit: int = 60, offset: int = 0) -> list[PublicProduct]:
    df = _frame().iloc[offset: offset + limit]
    return [_to_public(r) for r in df.itertuples()]


def get_product(product_id: str) -> PublicProduct | None:
    df = _frame()
    hit = df[df["product_id"] == product_id]
    if hit.empty:
        return None
    return _to_public(next(hit.itertuples()))


def get_economics(product_id: str) -> ProductEconomics | None:
    df = _frame()
    hit = df[df["product_id"] == product_id]
    if hit.empty:
        return None
    r = next(hit.itertuples())
    return ProductEconomics(
        product_id=str(r.product_id),
        selling_price=Decimal(str(r.selling_price)),
        cost_of_goods=Decimal(str(r.cost_of_goods)),
        base_shipping_cost=Decimal(str(r.base_shipping_cost)),
        historical_return_rate=Decimal(str(r.historical_return_rate)),
        inventory_level=int(r.inventory_level),
    )


def categories() -> list[str]:
    return sorted(_frame()["category"].unique().tolist())


def search(
    query: str | None = None,
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    in_stock_only: bool = True,
    limit: int = 5,
) -> list[PublicProduct]:
    """Deterministic catalog search.

    Ranking is explicit and inspectable — token overlap, then price ascending.
    No embedding model, because a buyer constraint like "under ₹6,000" is a
    filter, not a similarity question, and pretending otherwise would add a
    dependency without adding correctness.
    """
    df = _frame()

    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if max_price is not None:
        df = df[df["selling_price"] <= float(max_price)]
    if min_price is not None:
        df = df[df["selling_price"] >= float(min_price)]
    if in_stock_only:
        df = df[df["inventory_level"] > 0]

    if query:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        if tokens:
            hay = (
                df["name"].str.lower() + " "
                + df["category"].str.lower() + " "
                + df["subcategory"].str.lower()
            )
            df = df.assign(_score=[sum(t in h for t in tokens) for h in hay])
            df = df[df["_score"] > 0].sort_values(
                ["_score", "selling_price"], ascending=[False, True]
            )
        else:
            df = df.sort_values("selling_price")
    else:
        df = df.sort_values("selling_price")

    return [_to_public(r) for r in df.head(limit).itertuples()]