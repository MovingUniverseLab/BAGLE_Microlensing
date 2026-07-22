"""Google Drive helpers for the research notebook folder tree."""

from __future__ import annotations

from typing import Any

from research_notebook.config import load_config, save_config


FOLDER_MIME = "application/vnd.google-apps.folder"
DOC_MIME = "application/vnd.google-apps.document"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"


def find_child_by_name(
    drive: Any, parent_id: str, name: str, mime_type: str | None = None
) -> dict[str, Any] | None:
    """Find a Drive file by name under a parent.

    Parameters
    ----------
    drive :
        Drive API service.
    parent_id : str
        Parent folder ID.
    name : str
        Exact file name.
    mime_type : str, optional
        Restrict to this MIME type.

    Returns
    -------
    dict or None
        First matching file metadata, or None.
    """

    q = (
        f"name = '{_escape(name)}' and '{parent_id}' in parents "
        "and trashed = false"
    )
    if mime_type:
        q += f" and mimeType = '{mime_type}'"

    resp = (
        drive.files()
        .list(q=q, spaces="drive", fields="files(id, name, mimeType, webViewLink)")
        .execute()
    )
    files = resp.get("files", [])
    return files[0] if files else None


def create_folder(drive: Any, name: str, parent_id: str | None = None) -> dict:
    """Create a Drive folder.

    Parameters
    ----------
    drive :
        Drive API service.
    name : str
        Folder name.
    parent_id : str, optional
        Parent folder ID (My Drive root if omitted).

    Returns
    -------
    dict
        Created folder metadata.
    """

    meta: dict[str, Any] = {"name": name, "mimeType": FOLDER_MIME}
    if parent_id:
        meta["parents"] = [parent_id]
    return (
        drive.files()
        .create(body=meta, fields="id, name, webViewLink")
        .execute()
    )


def ensure_folder(
    drive: Any, name: str, parent_id: str | None = None
) -> dict[str, Any]:
    """Return existing folder or create it.

    Parameters
    ----------
    drive :
        Drive API service.
    name : str
        Folder name.
    parent_id : str, optional
        Parent folder ID.

    Returns
    -------
    dict
        Folder metadata with ``id``.
    """

    if parent_id:
        existing = find_child_by_name(drive, parent_id, name, FOLDER_MIME)
        if existing:
            return existing
    return create_folder(drive, name, parent_id)


def create_doc(drive: Any, name: str, parent_id: str) -> dict[str, Any]:
    """Create an empty Google Doc in a folder.

    Parameters
    ----------
    drive :
        Drive API service.
    name : str
        Document title.
    parent_id : str
        Parent folder ID.

    Returns
    -------
    dict
        File metadata.
    """

    meta = {
        "name": name,
        "mimeType": DOC_MIME,
        "parents": [parent_id],
    }
    return (
        drive.files()
        .create(body=meta, fields="id, name, webViewLink")
        .execute()
    )


def copy_file(
    drive: Any, file_id: str, name: str, parent_id: str
) -> dict[str, Any]:
    """Copy a Drive file into a folder.

    Parameters
    ----------
    drive :
        Drive API service.
    file_id : str
        Source file ID.
    name : str
        New name.
    parent_id : str
        Destination folder ID.

    Returns
    -------
    dict
        Copied file metadata.
    """

    body = {"name": name, "parents": [parent_id]}
    return (
        drive.files()
        .copy(fileId=file_id, body=body, fields="id, name, webViewLink")
        .execute()
    )


def create_sheet(drive: Any, name: str, parent_id: str) -> dict[str, Any]:
    """Create a Google Sheet in a folder.

    Parameters
    ----------
    drive :
        Drive API service.
    name : str
        Spreadsheet title.
    parent_id : str
        Parent folder ID.

    Returns
    -------
    dict
        File metadata.
    """

    meta = {
        "name": name,
        "mimeType": SHEET_MIME,
        "parents": [parent_id],
    }
    return (
        drive.files()
        .create(body=meta, fields="id, name, webViewLink")
        .execute()
    )


def upload_binary(
    drive: Any,
    local_path: str,
    name: str,
    parent_id: str,
    mime_type: str = "image/png",
) -> dict[str, Any]:
    """Upload a binary file (e.g. figure) to Drive.

    Parameters
    ----------
    drive :
        Drive API service.
    local_path : str
        Local file path.
    name : str
        Destination file name.
    parent_id : str
        Parent folder ID.
    mime_type : str, optional
        MIME type.

    Returns
    -------
    dict
        Uploaded file metadata including ``id`` and ``webViewLink``.
    """

    from googleapiclient.http import MediaFileUpload

    meta = {"name": name, "parents": [parent_id]}
    media = MediaFileUpload(local_path, mimetype=mime_type, resumable=True)
    return (
        drive.files()
        .create(
            body=meta,
            media_body=media,
            fields="id, name, webViewLink, webContentLink",
        )
        .execute()
    )


def make_file_public_readable(drive: Any, file_id: str) -> None:
    """Grant anyone-with-link reader access (needed for Docs inline images).

    Parameters
    ----------
    drive :
        Drive API service.
    file_id : str
        File ID to share.
    """

    drive.permissions().create(
        fileId=file_id,
        body={"type": "anyone", "role": "reader"},
        fields="id",
    ).execute()
    return None


def init_notebook_tree(docs: Any, drive: Any, sheets: Any) -> dict[str, Any]:
    """Create the Research Notebook Drive tree and templates.

    Parameters
    ----------
    docs :
        Docs API service.
    drive :
        Drive API service.
    sheets :
        Sheets API service.

    Returns
    -------
    dict
        Updated configuration with folder and template IDs.
    """

    cfg = load_config()
    root_name = cfg.get("notebook_root_name", "Research Notebook")

    # Prefer existing root by searching under My Drive via name if no id.
    root_id = cfg["folder_ids"].get("root")
    if root_id:
        root = {"id": root_id, "name": root_name}
    else:
        # Search for existing top-level folder with this name.
        q = (
            f"name = '{_escape(root_name)}' and mimeType = '{FOLDER_MIME}' "
            "and 'root' in parents and trashed = false"
        )
        resp = (
            drive.files()
            .list(q=q, spaces="drive", fields="files(id, name, webViewLink)")
            .execute()
        )
        files = resp.get("files", [])
        root = files[0] if files else create_folder(drive, root_name, None)

    cfg["folder_ids"]["root"] = root["id"]

    subfolders = {
        "templates": "00_Templates",
        "index": "01_Index",
        "daily": "02_Daily",
        "weekly": "03_Weekly",
        "monthly": "04_Monthly",
        "yearly": "05_Yearly",
        "projects": "06_Projects",
        "shared": "07_Shared",
        "assets": "08_Assets",
    }

    for key, name in subfolders.items():
        folder = ensure_folder(drive, name, root["id"])
        cfg["folder_ids"][key] = folder["id"]

    # Templates
    templates_id = cfg["folder_ids"]["templates"]
    daily_tmpl = _ensure_named_doc(
        drive, docs, templates_id, "Daily entry template", _daily_template_text()
    )
    weekly_tmpl = _ensure_named_doc(
        drive, docs, templates_id, "Weekly review template", _weekly_template_text()
    )
    monthly_tmpl = _ensure_named_doc(
        drive,
        docs,
        templates_id,
        "Monthly summary template",
        _monthly_template_text(),
    )
    cfg["template_ids"]["daily"] = daily_tmpl["id"]
    cfg["template_ids"]["weekly"] = weekly_tmpl["id"]
    cfg["template_ids"]["monthly"] = monthly_tmpl["id"]

    # Index sheet
    index_folder = cfg["folder_ids"]["index"]
    existing_index = find_child_by_name(drive, index_folder, "Index", SHEET_MIME)
    if existing_index:
        sheet_meta = existing_index
    else:
        sheet_meta = create_sheet(drive, "Index", index_folder)
        _init_index_headers(sheets, sheet_meta["id"])

    cfg["index_sheet_id"] = sheet_meta["id"]
    # Ensure headers even if sheet already existed empty.
    _init_index_headers(sheets, sheet_meta["id"])

    save_config(cfg)
    return cfg


def ensure_today_doc(
    drive: Any, docs: Any, cfg: dict[str, Any], date_str: str
) -> dict[str, Any]:
    """Ensure the daily Google Doc for ``date_str`` exists.

    Parameters
    ----------
    drive :
        Drive API service.
    docs :
        Docs API service.
    cfg : dict
        Loaded configuration.
    date_str : str
        Calendar date ``YYYY-MM-DD``.

    Returns
    -------
    dict
        File metadata with ``id`` and ``webViewLink``.
    """

    daily_folder = cfg["folder_ids"]["daily"]
    if not daily_folder:
        raise RuntimeError("Daily folder ID missing. Run: research-notebook init")

    existing = find_child_by_name(drive, daily_folder, date_str, DOC_MIME)
    if existing:
        # Refresh webViewLink if needed.
        if "webViewLink" not in existing:
            existing = (
                drive.files()
                .get(fileId=existing["id"], fields="id, name, webViewLink")
                .execute()
            )
        return existing

    template_id = cfg["template_ids"].get("daily")
    if template_id:
        created = copy_file(drive, template_id, date_str, daily_folder)
    else:
        created = create_doc(drive, date_str, daily_folder)
        _seed_doc_text(docs, created["id"], f"Research notebook — {date_str}\n\n")

    return created


def ensure_assets_day_folder(
    drive: Any, cfg: dict[str, Any], date_str: str
) -> dict[str, Any]:
    """Ensure ``08_Assets/YYYY-MM-DD`` exists.

    Parameters
    ----------
    drive :
        Drive API service.
    cfg : dict
        Configuration.
    date_str : str
        Calendar date.

    Returns
    -------
    dict
        Folder metadata.
    """

    assets = cfg["folder_ids"].get("assets")
    if not assets:
        raise RuntimeError("Assets folder ID missing. Run: research-notebook init")
    return ensure_folder(drive, date_str, assets)


def doc_url(file_id: str) -> str:
    """Return a Google Docs edit URL.

    Parameters
    ----------
    file_id : str
        Document ID.

    Returns
    -------
    str
        HTTPS URL.
    """

    return f"https://docs.google.com/document/d/{file_id}/edit"


def _ensure_named_doc(
    drive: Any,
    docs: Any,
    parent_id: str,
    name: str,
    body_text: str,
) -> dict[str, Any]:
    """Ensure a named Doc exists; seed text if newly created."""

    existing = find_child_by_name(drive, parent_id, name, DOC_MIME)
    if existing:
        return existing
    created = create_doc(drive, name, parent_id)
    _seed_doc_text(docs, created["id"], body_text)
    return created


def _seed_doc_text(docs: Any, doc_id: str, text: str) -> None:
    """Insert text at the start of an empty Doc."""

    docs.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {"insertText": {"location": {"index": 1}, "text": text}}
            ]
        },
    ).execute()
    return None


def _init_index_headers(sheets: Any, spreadsheet_id: str) -> None:
    """Write Index header row if sheet looks empty."""

    headers = [
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
    result = (
        sheets.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range="A1:M1")
        .execute()
    )
    values = result.get("values", [])
    if values and values[0]:
        return None

    sheets.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range="A1",
        valueInputOption="RAW",
        body={"values": [headers]},
    ).execute()
    return None


def _daily_template_text() -> str:
    return (
        "Research notebook — daily\n\n"
        "Append-only log. New entries are added at the end with HH:MM headings.\n\n"
    )


def _weekly_template_text() -> str:
    return (
        "Weekly review\n\n"
        "Wins / progress\n\n"
        "Decisions\n\n"
        "Carry-over\n\n"
        "Key daily notes\n\n"
    )


def _monthly_template_text() -> str:
    return (
        "Monthly summary\n\n"
        "Themes\n\n"
        "Milestones / papers / proposals\n\n"
        "Students mentored\n\n"
    )


def _escape(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")
