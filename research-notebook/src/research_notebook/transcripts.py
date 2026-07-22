"""Scan local Cursor agent transcripts for a calendar day."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

TIMESTAMP_RE = re.compile(
    r"<timestamp>(.*?)</timestamp>", re.IGNORECASE | re.DOTALL
)
USER_QUERY_RE = re.compile(
    r"<user_query>\s*(.*?)\s*</user_query>", re.IGNORECASE | re.DOTALL
)


@dataclass
class TranscriptSummary:
    """Summary of one parent Cursor agent transcript.

    Attributes
    ----------
    transcript_id : str
        UUID directory / file stem.
    project : str
        Cursor project slug under ``~/.cursor/projects``.
    path : str
        Absolute path to the JSONL file.
    first_query : str
        First user query text (truncated).
    start_time : datetime or None
        First parsed timestamp in the configured timezone.
    n_user_turns : int
        Number of user role turns.
    research_like : bool
        Whether it passed allow/block filters as research work.
    skip_reason : str
        Why it was skipped, if any.
    """

    transcript_id: str
    project: str
    path: str
    first_query: str = ""
    start_time: datetime | None = None
    n_user_turns: int = 0
    research_like: bool = True
    skip_reason: str = ""
    sample_assistant: str = ""


def scan_transcripts_for_day(
    cfg: dict[str, Any], date_str: str
) -> list[TranscriptSummary]:
    """Scan agent transcripts whose activity falls on ``date_str``.

    Parameters
    ----------
    cfg : dict
        Configuration (transcripts_root, allow/block lists, timezone).
    date_str : str
        ``YYYY-MM-DD`` in the notebook timezone.

    Returns
    -------
    list of TranscriptSummary
        Parent transcripts for that day (subagents skipped by default).
    """

    root = Path(cfg.get("transcripts_root", Path.home() / ".cursor" / "projects"))
    root = root.expanduser()
    tz = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))
    skip_sub = bool(cfg.get("reconcile_skip_subagents", True))
    allow = [a.lower() for a in (cfg.get("project_allowlist") or [])]
    block = [b.lower() for b in (cfg.get("project_blocklist") or [])]

    results: list[TranscriptSummary] = []
    if not root.exists():
        return results

    for jsonl in _iter_parent_transcripts(root, skip_subagents=skip_sub):
        project = _project_slug(jsonl, root)
        summary = _summarize_transcript(jsonl, project, tz)
        if summary is None:
            continue
        if summary.start_time is None:
            # Fall back to mtime date.
            mtime = datetime.fromtimestamp(jsonl.stat().st_mtime, tz=tz)
            summary.start_time = mtime
        if summary.start_time.strftime("%Y-%m-%d") != date_str:
            continue

        # Allow / block filtering.
        proj_l = project.lower()
        if block and any(b in proj_l for b in block):
            summary.research_like = False
            summary.skip_reason = "blocklist"
        elif allow and not any(a in proj_l for a in allow):
            summary.research_like = False
            summary.skip_reason = "not in allowlist"
        elif summary.n_user_turns == 0 or not summary.first_query.strip():
            summary.research_like = False
            summary.skip_reason = "empty"
        else:
            summary.research_like = True

        results.append(summary)

    results.sort(key=lambda s: s.start_time or datetime.min.replace(tzinfo=tz))
    return results


def _iter_parent_transcripts(
    root: Path, skip_subagents: bool = True
) -> Iterator[Path]:
    """Yield parent ``*.jsonl`` transcript files."""

    for path in root.rglob("*.jsonl"):
        parts = path.parts
        if skip_subagents and "subagents" in parts:
            continue
        # Prefer files named like <uuid>/<uuid>.jsonl
        if path.parent.name == path.stem or path.name.endswith(".jsonl"):
            yield path


def _project_slug(jsonl: Path, root: Path) -> str:
    """Return the project directory name under transcripts root."""

    try:
        rel = jsonl.relative_to(root)
        return rel.parts[0] if rel.parts else "unknown"
    except ValueError:
        return jsonl.parent.parent.name


def _summarize_transcript(
    path: Path, project: str, tz: ZoneInfo
) -> TranscriptSummary | None:
    """Parse a JSONL transcript into a short summary."""

    tid = path.stem
    first_query = ""
    first_ts: datetime | None = None
    n_user = 0
    sample_assistant = ""

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = obj.get("role")
                message = obj.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
                text = _content_to_text(content)

                if role == "user":
                    n_user += 1
                    ts = _parse_timestamp(text, tz)
                    if first_ts is None and ts is not None:
                        first_ts = ts
                    q = _extract_user_query(text)
                    if not first_query and q:
                        first_query = q[:500]
                elif role == "assistant" and not sample_assistant and text:
                    # Prefer short assistant prose without tool dumps.
                    cleaned = re.sub(r"\s+", " ", text).strip()
                    if cleaned and "tool_use" not in cleaned[:40]:
                        sample_assistant = cleaned[:400]
    except OSError:
        return None

    return TranscriptSummary(
        transcript_id=tid,
        project=project,
        path=str(path),
        first_query=first_query or "(no user_query found)",
        start_time=first_ts,
        n_user_turns=n_user,
        sample_assistant=sample_assistant,
    )


def _content_to_text(content: Any) -> str:
    """Flatten message content to a string."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _parse_timestamp(text: str, tz: ZoneInfo) -> datetime | None:
    """Parse the first ``<timestamp>`` from message text."""

    m = TIMESTAMP_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    # Example: Tuesday, Jul 21, 2026, 7:29 PM (UTC-7)
    for fmt in (
        "%A, %b %d, %Y, %I:%M %p (%Z)",
        "%A, %B %d, %Y, %I:%M %p (%Z)",
    ):
        try:
            # Strip timezone name in parens; use offset if present.
            cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", raw)
            dt = datetime.strptime(cleaned, fmt.replace(" (%Z)", ""))
            return dt.replace(tzinfo=tz)
        except ValueError:
            continue

    # Fallback: dateutil if available.
    try:
        from dateutil import parser as date_parser

        return date_parser.parse(raw).astimezone(tz)
    except Exception:
        return None


def _extract_user_query(text: str) -> str:
    """Extract ``<user_query>`` body if present."""

    m = USER_QUERY_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: strip timestamp tags.
    cleaned = TIMESTAMP_RE.sub("", text).strip()
    return cleaned[:500]
