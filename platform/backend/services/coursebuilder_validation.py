from __future__ import annotations


SUPPORTED_BLOCK_TYPES = {
    "explanation",
    "definition",
    "example",
    "table",
    "equation",
    "chart",
    "diagram",
    "procedure",
    "warning",
    "summary",
}


def normalize_block_type(value: str) -> str:
    block_type = str(value or "explanation").strip().lower()
    return block_type if block_type in SUPPORTED_BLOCK_TYPES else "explanation"


def support_status(citations: list[dict]) -> str:
    return "supported" if citations else "insufficient_source_material"


def insufficient_source_message() -> str:
    return (
        "The uploaded documents do not contain enough source material to teach "
        "this lesson reliably."
    )


def validate_chart_spec(data: dict) -> dict:
    chart_type = str(data.get("chart_type") or "bar").lower()
    if chart_type not in {"bar", "line", "pie"}:
        chart_type = "bar"
    rows = data.get("data")
    if not isinstance(rows, list):
        rows = []
    return {
        **data,
        "chart_type": chart_type,
        "data": rows,
        "x_key": str(data.get("x_key") or "label"),
        "y_keys": data.get("y_keys") if isinstance(data.get("y_keys"), list) else ["value"],
    }
