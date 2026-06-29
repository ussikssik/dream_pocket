from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

import pandas as pd


OrderId = Any
OrderLoader = Callable[[OrderId], pd.DataFrame]


class OrderDataLoader(Protocol):
    """Protocol for DB/parquet/CSV backed order loaders."""

    def __call__(self, order_id: OrderId) -> pd.DataFrame:
        """Return one order as a DataFrame containing label, metadata, and features."""


OrderList = Iterable[OrderId]
