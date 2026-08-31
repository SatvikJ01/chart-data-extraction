"""Submission assembly.

AUDIT NOTE (Phase 0, bug 6)
---------------------------
The original notebook post-processed the submission with chained pandas
assignment inside a Python loop::

    for i in range(len(sub_df)):
        sub_df['data_series'][i] = [...]

``sub_df['data_series'][i] = ...`` is chained indexing; pandas does not
guarantee it writes through to the frame, and it raises
``SettingWithCopyWarning`` (or silently no-ops) depending on version and on
whether the column is a view. It also ran a Python-level loop over every row.

Here the cleaning is a vectorised column operation applied before the frame is
built, so there is no chained assignment at all.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd

from chart_extraction.config import PipelineConfig


def _format_value(value: object) -> str:
    """Render one data-series element, mapping non-finite floats to '0.0'.

    The notebook filtered the *string* 'nan' after joining, which caught NaN but
    not inf/-inf. Filtering before the join catches both.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "0.0"
    return str(value)


def format_series(values: Sequence[object], placeholder: str) -> str:
    """Join one data series to the submission's ';'-separated format."""
    if values is None or len(values) == 0:
        return placeholder
    rendered = ";".join(_format_value(v) for v in values)
    return rendered if rendered.strip(";").strip() else placeholder


def build_submission(
    image_ids: Sequence[str],
    chart_types: Mapping[str, str],
    x_series: Mapping[str, Sequence[object]],
    y_series: Mapping[str, Sequence[object]],
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Build the submission frame, keyed on image id throughout.

    Every lookup is a dict access on ``image_id`` -- there is no positional
    alignment anywhere in this function (bug 3).
    """
    config = config or PipelineConfig()
    placeholder = config.placeholder_data_series

    rows = []
    for image_id in image_ids:
        chart_type = chart_types.get(image_id, config.placeholder_chart_type)
        rows.append(
            {
                "id": f"{image_id}_x",
                "data_series": format_series(x_series.get(image_id), placeholder),
                "chart_type": chart_type,
            }
        )
    for image_id in image_ids:
        chart_type = chart_types.get(image_id, config.placeholder_chart_type)
        rows.append(
            {
                "id": f"{image_id}_y",
                "data_series": format_series(y_series.get(image_id), placeholder),
                "chart_type": chart_type,
            }
        )

    return pd.DataFrame(rows, columns=["id", "data_series", "chart_type"])
