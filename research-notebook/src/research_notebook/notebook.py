"""Append entries and maintain the Index spreadsheet."""

from __future__ import annotations

import mimetypes
import socket
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from research_notebook import drive_ops
from research_notebook.docs_format import (
    build_append_requests,
    document_plain_text,
    get_document_end_index,
    prepare_entry_markdown,
)


INDEX_HEADERS = [
    "date",
    "time",
    "title",
    "project",
    "doc_url",
    "machine",
    "workspace",
    "repo",
    "commit",
    "transcript_id",
    "asset_folder_url",
    "status",
    "source",
]


def now_in_tz(tz_name: str) -> datetime:
    """Current time in the configured timezone.

    Parameters
    ----------
    tz_name : str
        IANA timezone name.

    Returns
    -------
    datetime.datetime
        Timezone-aware now.
    """

    return datetime.now(ZoneInfo(tz_name))


def append_entry(
    docs: Any,
    drive: Any,
    sheets: Any,
    cfg: dict[str, Any],
    *,
    title: str,
    body_md: str,
    project: str = "",
    workspace: str = "",
    repo: str = "",
    commit: str = "",
    transcript_id: str = "",
    figure_paths: list[str] | None = None,
    source: str = "ad-hoc",
    status: str = "logged",
    date_str: str | None = None,
    extra_meta: list[str] | None = None,
) -> dict[str, Any]:
    """Append a structured entry to today's daily Doc and Index.

    Parameters
    ----------
    docs, drive, sheets :
        Google API clients.
    cfg : dict
        Configuration.
    title : str
        Short title (time prefix added if missing).
    body_md : str
        Markdown body.
    project : str, optional
        Project tag.
    workspace : str, optional
        Absolute workspace path.
    repo : str, optional
        Repository name.
    commit : str, optional
        Git commit SHA.
    transcript_id : str, optional
        Cursor transcript UUID.
    figure_paths : list of str, optional
        Local figure paths to upload and embed.
    source : str, optional
        ``ad-hoc``, ``reconcile``, or ``github``.
    status : str, optional
        Status string for Index.
    date_str : str, optional
        Override calendar date.
    extra_meta : list of str, optional
        Extra metadata lines.

    Returns
    -------
    dict
        Result with ``doc_id``, ``doc_url``, ``date``, ``time``.
    """

    tz = cfg.get("timezone", "America/Los_Angeles")
    now = now_in_tz(tz)
    date_str = date_str or now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    if "–" not in title and "-" not in title[:6]:
        full_title = f"{time_str} – {title}"
    else:
        full_title = title

    day_doc = drive_ops.ensure_today_doc(drive, docs, cfg, date_str)
    doc_id = day_doc["id"]
    url = day_doc.get("webViewLink") or drive_ops.doc_url(doc_id)

    machine = cfg.get("machine_label") or socket.gethostname()
    meta_lines = [
        f"Machine: {machine} ({socket.gethostname()})",
    ]
    if workspace:
        meta_lines.append(f"Workspace: {workspace}")
    if repo or commit:
        meta_lines.append(
            f"Repos: {repo or '?'} @ {(commit[:12] if commit else '?')}"
        )
    if transcript_id:
        meta_lines.append(f"transcript-id: {transcript_id}")
    if extra_meta:
        meta_lines.extend(extra_meta)

    prepared = prepare_entry_markdown(
        full_title, body_md, meta_lines, figure_paths=figure_paths or []
    )

    # Upload figures to assets day folder.
    image_uris: dict[int, str] = {}
    asset_folder_url = ""
    if prepared.image_slots:
        assets_folder = drive_ops.ensure_assets_day_folder(drive, cfg, date_str)
        asset_folder_url = assets_folder.get("webViewLink", "")
        if not asset_folder_url:
            assets_folder = (
                drive.files()
                .get(fileId=assets_folder["id"], fields="id, webViewLink")
                .execute()
            )
            asset_folder_url = assets_folder.get("webViewLink", "")

        for slot in prepared.image_slots:
            local = Path(slot["local_path"]).expanduser()
            if not local.exists():
                continue
            mime, _ = mimetypes.guess_type(str(local))
            mime = mime or "application/octet-stream"
            uploaded = drive_ops.upload_binary(
                drive,
                str(local),
                local.name,
                assets_folder["id"],
                mime_type=mime,
            )
            # Docs insertInlineImage needs a URL Google can fetch.
            drive_ops.make_file_public_readable(drive, uploaded["id"])
            uri = f"https://drive.google.com/uc?id={uploaded['id']}"
            image_uris[slot["index"]] = uri
            slot["drive_id"] = uploaded["id"]
            slot["uri"] = uri

    # Re-fetch end index immediately before write (concurrent append safety).
    end_index = get_document_end_index(docs, doc_id)
    requests = build_append_requests(end_index, prepared, image_uris=image_uris)
    docs.documents().batchUpdate(
        documentId=doc_id, body={"requests": requests}
    ).execute()

    append_index_row(
        sheets,
        cfg,
        {
            "date": date_str,
            "time": time_str,
            "title": full_title,
            "project": project,
            "doc_url": url,
            "machine": machine,
            "workspace": workspace,
            "repo": repo,
            "commit": commit,
            "transcript_id": transcript_id,
            "asset_folder_url": asset_folder_url,
            "status": status,
            "source": source,
        },
    )

    return {
        "doc_id": doc_id,
        "doc_url": url,
        "date": date_str,
        "time": time_str,
        "title": full_title,
    }


def append_index_row(sheets: Any, cfg: dict[str, Any], row: dict[str, str]) -> None:
    """Append one row to the Index sheet.

    Parameters
    ----------
    sheets :
        Sheets API service.
    cfg : dict
        Configuration with ``index_sheet_id``.
    row : dict
        Column values keyed by header name.
    """

    sheet_id = cfg.get("index_sheet_id")
    if not sheet_id:
        raise RuntimeError("index_sheet_id missing. Run: research-notebook init")

    values = [[row.get(h, "") for h in INDEX_HEADERS]]
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range="A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()
    return None


def read_index_rows(sheets: Any, cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Read all Index rows as dicts.

    Parameters
    ----------
    sheets :
        Sheets API service.
    cfg : dict
        Configuration.

    Returns
    -------
    list of dict
        One dict per data row.
    """

    sheet_id = cfg.get("index_sheet_id")
    if not sheet_id:
        return []

    result = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=sheet_id, range="A1:M")
        .execute()
    )
    values = result.get("values", [])
    if len(values) < 2:
        return []

    headers = values[0]
    rows: list[dict[str, str]] = []
    for raw in values[1:]:
        padded = raw + [""] * (len(headers) - len(raw))
        rows.append({headers[i]: padded[i] for i in range(len(headers))})
    return rows


def index_coverage_for_date(
    sheets: Any, cfg: dict[str, Any], date_str: str
) -> dict[str, set[str]]:
    """Collect transcript IDs and commit SHAs already logged for a date.

    Parameters
    ----------
    sheets :
        Sheets API service.
    cfg : dict
        Configuration.
    date_str : str
        ``YYYY-MM-DD``.

    Returns
    -------
    dict
        Keys ``transcript_ids`` and ``commits`` with sets of strings.
    """

    tids: set[str] = set()
    commits: set[str] = set()
    for row in read_index_rows(sheets, cfg):
        if row.get("date") != date_str:
            continue
        if row.get("transcript_id"):
            tids.add(row["transcript_id"].strip())
        if row.get("commit"):
            commits.add(row["commit"].strip())
    return {"transcript_ids": tids, "commits": commits}


def today_doc_text(
    docs: Any, drive: Any, cfg: dict[str, Any], date_str: str
) -> str:
    """Return plain text of the daily Doc (empty string if missing).

    Parameters
    ----------
    docs, drive :
        API clients.
    cfg : dict
        Configuration.
    date_str : str
        Date string.

    Returns
    -------
    str
        Document text.
    """

    daily_folder = cfg["folder_ids"].get("daily")
    if not daily_folder:
        return ""
    existing = drive_ops.find_child_by_name(
        drive, daily_folder, date_str, drive_ops.DOC_MIME
    )
    if not existing:
        return ""
    return document_plain_text(docs, existing["id"])
