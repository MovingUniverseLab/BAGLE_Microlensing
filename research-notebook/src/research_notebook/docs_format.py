"""Convert entry content into well-styled Google Docs batchUpdate requests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

BlockKind = Literal[
    "title",  # day or entry heading
    "h2",
    "h3",
    "meta",
    "body",
    "bullet",
    "code",
    "rule",
    "blank",
    "image",
    "link_line",
]


@dataclass
class TextBlock:
    """One paragraph (or image placeholder) to insert.

    Attributes
    ----------
    kind : str
        Block kind controlling Docs paragraph/text style.
    text : str
        Paragraph text (no trailing newline required).
    image_index : int, optional
        For ``kind='image'``, index into uploaded image URIs.
    """

    kind: BlockKind
    text: str = ""
    image_index: int | None = None


@dataclass
class PreparedEntry:
    """Structured blocks ready for Docs insertion.

    Attributes
    ----------
    blocks : list of TextBlock
        Ordered content blocks.
    image_slots : list of dict
        Figure upload slots with ``local_path`` / ``index``.
    """

    blocks: list[TextBlock] = field(default_factory=list)
    image_slots: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Flatten blocks to plain text (for tests / debugging)."""

        lines: list[str] = []
        for b in self.blocks:
            if b.kind == "blank":
                lines.append("")
            elif b.kind == "image":
                lines.append(f"[[IMAGE:{b.image_index}]]")
            elif b.kind == "rule":
                lines.append("")
            else:
                lines.append(b.text)
        return "\n".join(lines) + "\n"


def prepare_entry_markdown(
    title: str,
    body_md: str,
    meta_lines: list[str],
    figure_paths: list[str] | None = None,
) -> PreparedEntry:
    """Build structured blocks from title, metadata, markdown, and figures.

    Parameters
    ----------
    title : str
        Entry heading like ``14:32 – bagle GP residuals``.
    body_md : str
        Markdown body.
    meta_lines : list of str
        Metadata lines (Machine, Workspace, …).
    figure_paths : list of str, optional
        Local figure paths to embed.

    Returns
    -------
    PreparedEntry
        Structured blocks plus image slots.
    """

    figure_paths = figure_paths or []
    blocks: list[TextBlock] = [TextBlock("h2", title.strip())]

    for line in meta_lines:
        if line.strip():
            blocks.append(TextBlock("meta", line.strip()))

    blocks.append(TextBlock("blank"))
    blocks.extend(_markdown_to_blocks(body_md))

    image_slots: list[dict[str, Any]] = []
    if figure_paths:
        blocks.append(TextBlock("h3", "Figures"))
        for i, path in enumerate(figure_paths):
            image_slots.append({"marker": f"[[IMAGE:{i}]]", "local_path": path, "index": i})
            blocks.append(TextBlock("image", image_index=i))
        blocks.append(TextBlock("blank"))

    blocks.append(TextBlock("rule"))
    blocks.append(TextBlock("blank"))
    return PreparedEntry(blocks=blocks, image_slots=image_slots)


def prepare_day_header(date_str: str) -> list[TextBlock]:
    """Return blocks for the top of a daily Doc.

    Parameters
    ----------
    date_str : str
        ``YYYY-MM-DD``.

    Returns
    -------
    list of TextBlock
        Title and short intro.
    """

    return [
        TextBlock("title", f"Research notebook — {date_str}"),
        TextBlock("body", "Append-only daily log. Entries use HH:MM headings."),
        TextBlock("blank"),
    ]


def _markdown_to_blocks(md: str) -> list[TextBlock]:
    """Parse a small Markdown subset into TextBlocks."""

    blocks: list[TextBlock] = []
    in_fence = False
    code_lines: list[str] = []

    for line in md.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_fence:
                blocks.append(TextBlock("code", "\n".join(code_lines)))
                code_lines = []
                in_fence = False
                blocks.append(TextBlock("blank"))
            else:
                in_fence = True
                code_lines = []
            continue

        if in_fence:
            code_lines.append(line)
            continue

        if not stripped:
            if blocks and blocks[-1].kind != "blank":
                blocks.append(TextBlock("blank"))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            blocks.append(TextBlock("h2" if level <= 2 else "h3", text))
            continue

        m = re.match(r"^[-*+]\s+(.*)$", stripped)
        if m:
            blocks.append(TextBlock("bullet", _strip_md_inline(m.group(1))))
            continue

        m = re.match(r"^\d+\.\s+(.*)$", stripped)
        if m:
            blocks.append(TextBlock("bullet", _strip_md_inline(m.group(1))))
            continue

        # Standalone URL or "label (url)" → link_line for styling.
        url_m = re.search(r"(https?://\S+)", stripped)
        if url_m and stripped.startswith("http"):
            blocks.append(TextBlock("link_line", stripped.rstrip(").,;")))
            continue

        blocks.append(TextBlock("body", _strip_md_inline(stripped)))

    if in_fence and code_lines:
        blocks.append(TextBlock("code", "\n".join(code_lines)))

    # Drop trailing blanks.
    while blocks and blocks[-1].kind == "blank":
        blocks.pop()
    return blocks


def _strip_md_inline(text: str) -> str:
    """Remove simple bold/italic/code markers."""

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


def build_append_requests(
    end_index: int,
    prepared: PreparedEntry,
    image_uris: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Build Docs batchUpdate requests to append styled blocks at document end.

    Parameters
    ----------
    end_index : int
        Current document end index.
    prepared : PreparedEntry
        Structured entry.
    image_uris : dict, optional
        Map image slot index → fetchable URI.

    Returns
    -------
    list of dict
        Docs API requests (inserts then styles; images last-to-first).
    """

    image_uris = image_uris or {}
    insert_at = max(1, end_index - 1)
    return _blocks_to_requests(insert_at, prepared.blocks, image_uris)


def build_replace_body_requests(
    end_index: int,
    blocks: list[TextBlock],
    image_uris: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Clear the document body and write ``blocks`` from the start.

    Parameters
    ----------
    end_index : int
        Current document end index.
    blocks : list of TextBlock
        Full-day content including header.
    image_uris : dict, optional
        Image URI map.

    Returns
    -------
    list of dict
        Docs API requests.
    """

    image_uris = image_uris or {}
    requests: list[dict[str, Any]] = []
    # Delete existing body content (keep final newline sentinel).
    if end_index > 2:
        requests.append(
            {
                "deleteContentRange": {
                    "range": {"startIndex": 1, "endIndex": end_index - 1}
                }
            }
        )
    requests.extend(_blocks_to_requests(1, blocks, image_uris))
    return requests


def _blocks_to_requests(
    insert_at: int,
    blocks: list[TextBlock],
    image_uris: dict[int, str],
) -> list[dict[str, Any]]:
    """Insert block text then apply paragraph/text styles and images."""

    # Build contiguous text with paragraph boundaries tracked.
    parts: list[str] = []
    spans: list[tuple[int, int, TextBlock]] = []  # start, end, block within inserted text
    cursor = 0
    image_positions: list[tuple[int, int, int]] = []  # start, end, image_index

    for block in blocks:
        if block.kind == "blank":
            chunk = "\n"
            start = cursor
            parts.append(chunk)
            cursor += len(chunk)
            spans.append((start, cursor, block))
            continue

        if block.kind == "rule":
            # Use a short horizontal-rule-like line; Docs has no true HR via API easily.
            chunk = "────────\n"
            start = cursor
            parts.append(chunk)
            cursor += len(chunk)
            spans.append((start, cursor, block))
            continue

        if block.kind == "image":
            marker = f"[[IMAGE:{block.image_index}]]"
            chunk = marker + "\n"
            start = cursor
            parts.append(chunk)
            cursor += len(chunk)
            spans.append((start, cursor, block))
            if block.image_index is not None:
                image_positions.append((start, start + len(marker), block.image_index))
            continue

        if block.kind == "code":
            # Keep code as one paragraph with internal newlines replaced by soft breaks
            # via multiple paragraphs for readability.
            code_text = block.text.rstrip("\n")
            for i, line in enumerate(code_text.split("\n") or [""]):
                chunk = (line if line else " ") + "\n"
                start = cursor
                parts.append(chunk)
                cursor += len(chunk)
                spans.append((start, cursor, TextBlock("code", line)))
            continue

        chunk = block.text.rstrip("\n") + "\n"
        start = cursor
        parts.append(chunk)
        cursor += len(chunk)
        spans.append((start, cursor, block))

    text = "".join(parts)
    if not text:
        return []

    requests: list[dict[str, Any]] = [
        {"insertText": {"location": {"index": insert_at}, "text": text}}
    ]

    # Apply styles (absolute indices = insert_at + relative).
    for start, end, block in spans:
        abs_start = insert_at + start
        abs_end = insert_at + end
        if abs_end <= abs_start:
            continue

        named = _named_style(block.kind)
        if named:
            requests.append(
                {
                    "updateParagraphStyle": {
                        "range": {"startIndex": abs_start, "endIndex": abs_end},
                        "paragraphStyle": {"namedStyleType": named},
                        "fields": "namedStyleType",
                    }
                }
            )

        if block.kind == "bullet":
            requests.append(
                {
                    "createParagraphBullets": {
                        "range": {"startIndex": abs_start, "endIndex": abs_end},
                        "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                    }
                }
            )

        if block.kind == "code":
            requests.append(
                {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": abs_start,
                            "endIndex": max(abs_start + 1, abs_end - 1),
                        },
                        "textStyle": {
                            "weightedFontFamily": {"fontFamily": "Roboto Mono"},
                            "fontSize": {"magnitude": 9, "unit": "PT"},
                        },
                        "fields": "weightedFontFamily,fontSize",
                    }
                }
            )

        if block.kind == "meta":
            # Bold the label before the first colon.
            label_end = block.text.find(":")
            if label_end > 0:
                requests.append(
                    {
                        "updateTextStyle": {
                            "range": {
                                "startIndex": abs_start,
                                "endIndex": abs_start + label_end + 1,
                            },
                            "textStyle": {"bold": True},
                            "fields": "bold",
                        }
                    }
                )
            requests.append(
                {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": abs_start,
                            "endIndex": max(abs_start + 1, abs_end - 1),
                        },
                        "textStyle": {
                            "fontSize": {"magnitude": 10, "unit": "PT"},
                            "foregroundColor": {
                                "color": {
                                    "rgbColor": {
                                        "red": 0.35,
                                        "green": 0.35,
                                        "blue": 0.35,
                                    }
                                }
                            },
                        },
                        "fields": "fontSize,foregroundColor",
                    }
                }
            )

        if block.kind == "link_line" or (
            block.kind == "bullet" and "https://" in block.text
        ):
            for m in re.finditer(r"https://\S+", block.text):
                url = m.group(0).rstrip(").,;")
                requests.append(
                    {
                        "updateTextStyle": {
                            "range": {
                                "startIndex": abs_start + m.start(),
                                "endIndex": abs_start + m.start() + len(url),
                            },
                            "textStyle": {
                                "link": {"url": url},
                                "foregroundColor": {
                                    "color": {
                                        "rgbColor": {
                                            "red": 0.06,
                                            "green": 0.33,
                                            "blue": 0.80,
                                        }
                                    }
                                },
                            },
                            "fields": "link,foregroundColor",
                        }
                    }
                )

    # Replace image markers from end to start.
    for start, end, img_idx in sorted(image_positions, key=lambda t: t[0], reverse=True):
        uri = image_uris.get(img_idx)
        if not uri:
            continue
        abs_start = insert_at + start
        abs_end = insert_at + end
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
                        "height": {"magnitude": 280, "unit": "PT"},
                        "width": {"magnitude": 400, "unit": "PT"},
                    },
                }
            }
        )

    return requests


def _named_style(kind: BlockKind) -> str | None:
    """Map block kind to Docs named paragraph style."""

    if kind == "title":
        return "HEADING_1"
    if kind == "h2":
        return "HEADING_2"
    if kind == "h3":
        return "HEADING_3"
    return None


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
    content = doc.get("body", {}).get("content", [])
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
