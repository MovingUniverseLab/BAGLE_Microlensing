"""Convert a small Markdown subset into Google Docs batchUpdate requests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedEntry:
    """Text and image placeholders ready for Docs insertion.

    Attributes
    ----------
    text : str
        Plain text with blank lines; image markers use ``[[IMAGE:n]]``.
    image_slots : list of dict
        Each dict has ``marker``, ``local_path``, and later ``uri`` / ``drive_id``.
    """

    text: str
    image_slots: list[dict[str, Any]] = field(default_factory=list)


def prepare_entry_markdown(
    title: str,
    body_md: str,
    meta_lines: list[str],
    figure_paths: list[str] | None = None,
) -> PreparedEntry:
    """Build append text from title, metadata, markdown body, and figures.

    Parameters
    ----------
    title : str
        Heading like ``14:32 – bagle GP residuals``.
    body_md : str
        Markdown body (headings, bullets, fenced code supported loosely).
    meta_lines : list of str
        Pre-formatted metadata lines (Machine, Workspace, …).
    figure_paths : list of str, optional
        Local figure paths to embed after Results-ish content.

    Returns
    -------
    PreparedEntry
        Text plus image slots.
    """

    figure_paths = figure_paths or []
    parts: list[str] = [f"{title}\n"]
    for line in meta_lines:
        parts.append(f"{line}\n")
    parts.append("\n")
    parts.append(_markdown_to_plain(body_md))
    parts.append("\n")

    image_slots: list[dict[str, Any]] = []
    if figure_paths:
        parts.append("Figures\n")
        for i, path in enumerate(figure_paths):
            marker = f"[[IMAGE:{i}]]"
            image_slots.append({"marker": marker, "local_path": path, "index": i})
            parts.append(f"{marker}\n")
        parts.append("\n")

    text = "".join(parts)
    if not text.endswith("\n"):
        text += "\n"
    # Trailing separator between entries.
    text += "────────────────────────────────────────\n\n"
    return PreparedEntry(text=text, image_slots=image_slots)


def _markdown_to_plain(md: str) -> str:
    """Best-effort Markdown → plain text for Docs insertText.

    Parameters
    ----------
    md : str
        Markdown source.

    Returns
    -------
    str
        Plain text with structure preserved as lines.
    """

    out: list[str] = []
    in_fence = False
    for line in md.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line.replace("```", "").strip() and line or "")
            continue
        if in_fence:
            out.append(line)
            continue

        # Headings → uppercase-ish plain headings.
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            out.append(m.group(2).strip())
            out.append("")
            continue

        # Bullets.
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m:
            out.append(f"• {m.group(2)}")
            continue

        # Numbered.
        m = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if m:
            out.append(f"• {m.group(2)}")
            continue

        # Strip simple bold/italic markers.
        cleaned = re.sub(r"[*`_]", "", line)
        out.append(cleaned)

    return "\n".join(out).rstrip() + "\n"


def build_append_requests(
    end_index: int,
    prepared: PreparedEntry,
    image_uris: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Build Docs ``batchUpdate`` requests to append at document end.

    Parameters
    ----------
    end_index : int
        Current document end index (exclusive body end).
    prepared : PreparedEntry
        Prepared text and image slots.
    image_uris : dict, optional
        Map from image slot index → publicly fetchable URI.

    Returns
    -------
    list of dict
        Docs API requests.
    """

    image_uris = image_uris or {}
    # Insert before the final newline of the doc body.
    insert_at = max(1, end_index - 1)
    text = prepared.text
    requests: list[dict[str, Any]] = [
        {"insertText": {"location": {"index": insert_at}, "text": text}}
    ]

    # After insert, markers sit at insert_at + offset.
    # Replace each [[IMAGE:n]] with an inline image (delete marker text, insert image).
    # Process from end to start so indices stay valid.
    slots_sorted = sorted(
        prepared.image_slots, key=lambda s: text.rfind(s["marker"]), reverse=True
    )
    for slot in slots_sorted:
        marker = slot["marker"]
        pos = text.find(marker)
        if pos < 0:
            continue
        uri = image_uris.get(slot["index"])
        if not uri:
            continue
        abs_start = insert_at + pos
        abs_end = abs_start + len(marker)
        requests.append(
            {
                "deleteContentRange": {
                    "range": {"startIndex": abs_start, "endIndex": abs_end}
                }
            }
        )
        requests.append(
            {
                "insertInlineImage": {
                    "location": {"index": abs_start},
                    "uri": uri,
                    "objectSize": {
                        "height": {"magnitude": 320, "unit": "PT"},
                        "width": {"magnitude": 420, "unit": "PT"},
                    },
                }
            }
        )

    # Style the first line (title) as HEADING_2.
    title_line = prepared.text.split("\n", 1)[0]
    title_end = insert_at + len(title_line)
    requests.append(
        {
            "updateParagraphStyle": {
                "range": {
                    "startIndex": insert_at,
                    "endIndex": title_end + 1,
                },
                "paragraphStyle": {"namedStyleType": "HEADING_2"},
                "fields": "namedStyleType",
            }
        }
    )

    return requests


def get_document_end_index(docs: Any, doc_id: str) -> int:
    """Return the end index of the document body.

    Parameters
    ----------
    docs :
        Docs API service.
    doc_id : str
        Document ID.

    Returns
    -------
    int
        End index.
    """

    doc = docs.documents().get(documentId=doc_id).execute()
    body = doc.get("body", {})
    content = body.get("content", [])
    if not content:
        return 1
    return int(content[-1]["endIndex"])


def document_plain_text(docs: Any, doc_id: str) -> str:
    """Extract plain text from a Google Doc.

    Parameters
    ----------
    docs :
        Docs API service.
    doc_id : str
        Document ID.

    Returns
    -------
    str
        Concatenated text.
    """

    doc = docs.documents().get(documentId=doc_id).execute()
    chunks: list[str] = []
    for el in doc.get("body", {}).get("content", []):
        para = el.get("paragraph")
        if not para:
            continue
        for pe in para.get("elements", []):
            text_run = pe.get("textRun")
            if text_run and "content" in text_run:
                chunks.append(text_run["content"])
    return "".join(chunks)
