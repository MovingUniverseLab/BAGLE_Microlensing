"""Google OAuth helpers for Docs / Drive / Sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from research_notebook.config import CLIENT_SECRET_PATH, TOKEN_PATH, ensure_config_dir

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_credentials(interactive: bool = True) -> Credentials:
    """Load or refresh OAuth credentials.

    Order of preference:

    1. ``~/.config/research_notebook/token.json`` from ``research-notebook auth``
    2. Interactive Desktop OAuth using ``client_secret.json``
    3. Application Default Credentials (e.g. ``gcloud auth application-default login``)
       if they already include the Docs/Drive/Sheets scopes

    Parameters
    ----------
    interactive : bool, optional
        If True and no valid token exists, run the browser OAuth flow.

    Returns
    -------
    google.oauth2.credentials.Credentials
        Authorized credentials.

    Raises
    ------
    FileNotFoundError
        If ``client_secret.json`` is missing and ADC is unavailable.
    RuntimeError
        If credentials cannot be obtained without interaction.
    """

    ensure_config_dir()
    creds: Credentials | None = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(creds)
        return creds

    # Try Application Default Credentials before requiring client_secret.
    adc = _try_adc()
    if adc is not None:
        return adc

    if not interactive:
        raise RuntimeError(
            "No valid Google token. Run: research-notebook auth\n"
            "Or: gcloud auth application-default login "
            "--scopes=" + ",".join(SCOPES)
        )

    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"Missing OAuth client secret at {CLIENT_SECRET_PATH}.\n"
            "See README.md / SETUP_OAUTH.md, or run:\n"
            "  gcloud auth application-default login --scopes="
            + ",".join(SCOPES)
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CLIENT_SECRET_PATH), SCOPES
    )
    creds = flow.run_local_server(port=0)
    _save_token(creds)
    return creds


def _try_adc() -> Any | None:
    """Return ADC credentials if they can authorize Docs/Drive/Sheets.

    Returns
    -------
    credentials or None
        Google auth credentials, or None if ADC is missing/insufficient.
    """

    try:
        import google.auth

        creds, _project = google.auth.default(scopes=SCOPES)
        if creds is None:
            return None
        if hasattr(creds, "refresh"):
            try:
                creds.refresh(Request())
            except Exception:
                # Some ADC types are valid without refresh.
                pass
        return creds
    except Exception:
        return None


def _save_token(creds: Credentials) -> None:
    """Persist credentials to token.json."""

    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return None


def build_services(
    interactive: bool = True,
) -> tuple[Any, Any, Any]:
    """Build Docs, Drive, and Sheets API clients.

    Parameters
    ----------
    interactive : bool, optional
        Passed to ``get_credentials``.

    Returns
    -------
    tuple
        ``(docs, drive, sheets)`` service objects.
    """

    creds = get_credentials(interactive=interactive)
    docs = build("docs", "v1", credentials=creds, cache_discovery=False)
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return docs, drive, sheets


def write_oauth_setup_instructions(path: Path | None = None) -> Path:
    """Write a short OAuth setup guide next to the config dir.

    Parameters
    ----------
    path : pathlib.Path, optional
        Destination path. Defaults to ``SETUP_OAUTH.md`` in the config dir.

    Returns
    -------
    pathlib.Path
        Path written.
    """

    ensure_config_dir()
    dest = path or (CLIENT_SECRET_PATH.parent / "SETUP_OAUTH.md")
    dest.write_text(
        """# Google OAuth setup for research-notebook

## Option A — Desktop OAuth client (recommended)

1. Open https://console.cloud.google.com/ and create (or select) a project,
   e.g. `mulab-bagle` or `research-notebook`.
2. Enable APIs: **Google Docs API**, **Google Drive API**, **Google Sheets API**.
3. Configure OAuth consent screen (External or Internal for Berkeley Workspace).
   Add your Google account as a test user if the app is in testing.
4. Create credentials → **OAuth client ID** → Application type **Desktop app**.
5. Download the JSON and save it as:
   `~/.config/research_notebook/client_secret.json`
6. Run: `research-notebook auth`
7. Copy `token.json` (and `config.yaml` after `init`) to other machines
   (laptop ↔ dragon) instead of re-doing the browser flow on headless hosts.

## Option B — gcloud Application Default Credentials

If you already use `gcloud` as `jlu.astro@berkeley.edu`:

```bash
gcloud auth application-default login \\
  --scopes=https://www.googleapis.com/auth/documents,https://www.googleapis.com/auth/drive,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/cloud-platform
```

Then `research-notebook init` can use ADC without `client_secret.json`.

Never commit `client_secret.json` or `token.json`.
""",
        encoding="utf-8",
    )
    return dest


def client_secret_status() -> dict[str, bool]:
    """Return whether client secret and token files exist.

    Returns
    -------
    dict
        Flags for ``client_secret`` and ``token``.
    """

    return {
        "client_secret": CLIENT_SECRET_PATH.exists(),
        "token": TOKEN_PATH.exists(),
    }
