from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SQLOderLoader:
    """Simple SQL order loader.

    The query template must contain ``{order_id}``. Prefer parameterized readers in
    production if order IDs come from users.
    """

    connection: Any
    query_template: str

    def __call__(self, order_id: Any) -> pd.DataFrame:
        query = self.query_template.format(order_id=order_id)
        return pd.read_sql_query(query, self.connection)


SQLOrderLoader = SQLOderLoader
