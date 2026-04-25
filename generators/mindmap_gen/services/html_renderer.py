import asyncio
import json
import uuid
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..schemas import MindMap

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


async def render(
    mindmap: MindMap,
    markdown: str,
    artifacts_dir: str,
) -> tuple[str, str]:
    """Render artifacts: standalone Markmap HTML + a JSON payload for inline render.

    The JSON file is what the platform's `MindmapRenderer` fetches to render
    the mind map inline in chat. The HTML is a self-contained, downloadable
    viewer (offline / sharing).

    Returns (json_path, html_path) — absolute filesystem paths.
    """
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uid = uuid.uuid4().hex[:12]
    base = f"mindmap_{uid}"
    json_path = out_dir / f"{base}.json"
    html_path = out_dir / f"{base}.html"

    template = _env.get_template("mindmap.html.jinja")
    html = template.render(
        title=mindmap.central_topic,
        markdown=markdown,
        filename=base,
    )
    payload = json.dumps(
        {
            "markdown": markdown,
            "central_topic": mindmap.central_topic,
            "main_branches": [b.text for b in mindmap.branches],
        },
        ensure_ascii=False,
    )

    await asyncio.gather(
        asyncio.to_thread(_write, json_path, payload),
        asyncio.to_thread(_write, html_path, html),
    )

    return str(json_path), str(html_path)
