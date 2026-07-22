"""Command-line interface for research-notebook."""

from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from research_notebook import __version__
from research_notebook.auth import (
    build_services,
    client_secret_status,
    get_credentials,
    write_oauth_setup_instructions,
)
from research_notebook.config import (
    CLIENT_SECRET_PATH,
    CONFIG_DIR,
    CONFIG_PATH,
    load_config,
    save_config,
)
from research_notebook import drive_ops
from research_notebook.github_commits import fetch_commits_for_day, format_github_rollup
from research_notebook.notebook import append_entry, now_in_tz
from research_notebook.reconcile import apply_reconcile, build_reconcile_report
from research_notebook.reformat import reformat_day_doc


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``research-notebook``.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(
        prog="research-notebook",
        description="Append research notes to native Google Docs",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth", help="Run Google OAuth and save token.json")
    sub.add_parser(
        "setup-oauth-docs",
        help="Write SETUP_OAUTH.md into the config directory",
    )
    p_init = sub.add_parser("init", help="Create Drive tree + templates + Index")
    p_init.add_argument(
        "--machine-label",
        default=None,
        help="Override machine_label in config",
    )

    p_ensure = sub.add_parser("ensure-today", help="Ensure today's daily Doc exists")
    p_ensure.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")

    p_link = sub.add_parser("link", help="Print today's Doc URL")
    p_link.add_argument("--date", default=None)

    p_append = sub.add_parser("append", help="Append an entry to today's Doc")
    p_append.add_argument("--title", required=True)
    p_append.add_argument("--body-file", required=True, help="Markdown body file")
    p_append.add_argument("--project", default="")
    p_append.add_argument("--workspace", default="")
    p_append.add_argument("--repo", default="")
    p_append.add_argument("--commit", default="")
    p_append.add_argument("--transcript-id", default="")
    p_append.add_argument(
        "--figure", action="append", default=[], help="Local figure path (repeatable)"
    )
    p_append.add_argument("--source", default="ad-hoc")
    p_append.add_argument("--date", default=None)

    p_rec = sub.add_parser(
        "reconcile-day",
        help="Scan Cursor transcripts + GitHub; report or backfill gaps",
    )
    p_rec.add_argument("--date", default=None)
    p_rec.add_argument("--dry-run", action="store_true")
    p_rec.add_argument("--append-missing", action="store_true")
    p_rec.add_argument("--skip-github", action="store_true")
    p_rec.add_argument("--skip-transcripts", action="store_true")
    p_rec.add_argument(
        "--projects",
        default=None,
        help="Comma-separated transcript project allowlist override",
    )

    p_gh = sub.add_parser("github-day", help="List today's GitHub commits")
    p_gh.add_argument("--date", default=None)

    p_reformat = sub.add_parser(
        "reformat-day",
        help="Rewrite a daily Doc with clean headings, bullets, and sections",
    )
    p_reformat.add_argument("--date", default=None)

    sub.add_parser("status", help="Show config / auth status")

    args = parser.parse_args(argv)

    if args.cmd == "setup-oauth-docs":
        path = write_oauth_setup_instructions()
        print(f"Wrote {path}")
        print(f"Place client secret at: {CLIENT_SECRET_PATH}")
        return 0

    if args.cmd == "auth":
        write_oauth_setup_instructions()
        status = client_secret_status()
        if not status["client_secret"]:
            print(
                f"Missing {CLIENT_SECRET_PATH}\n"
                f"Follow {CONFIG_DIR / 'SETUP_OAUTH.md'} then re-run auth.",
                file=sys.stderr,
            )
            return 1
        get_credentials(interactive=True)
        print(f"Saved token to {CONFIG_DIR / 'token.json'}")
        return 0

    if args.cmd == "status":
        return _cmd_status()

    # Remaining commands need Google APIs (except github-day can work without).
    if args.cmd == "github-day":
        return _cmd_github_day(args)

    try:
        docs, drive, sheets = build_services(interactive=False)
    except Exception as exc:
        print(f"Google auth error: {exc}", file=sys.stderr)
        print("Run: research-notebook auth", file=sys.stderr)
        return 1

    if args.cmd == "init":
        return _cmd_init(docs, drive, sheets, args)
    if args.cmd == "ensure-today":
        return _cmd_ensure_today(docs, drive, args)
    if args.cmd == "link":
        return _cmd_link(docs, drive, args)
    if args.cmd == "append":
        return _cmd_append(docs, drive, sheets, args)
    if args.cmd == "reconcile-day":
        return _cmd_reconcile(docs, drive, sheets, args)
    if args.cmd == "reformat-day":
        return _cmd_reformat(docs, drive, sheets, args)

    print(f"Unknown command: {args.cmd}", file=sys.stderr)
    return 2


def _date_or_today(cfg: dict, date_str: str | None) -> str:
    if date_str:
        return date_str
    return now_in_tz(cfg.get("timezone", "America/Los_Angeles")).strftime("%Y-%m-%d")


def _cmd_status() -> int:
    cfg = load_config()
    st = client_secret_status()
    print(f"config_dir: {CONFIG_DIR}")
    print(f"config: {CONFIG_PATH} ({'exists' if CONFIG_PATH.exists() else 'missing'})")
    print(f"client_secret: {'yes' if st['client_secret'] else 'NO'}")
    print(f"token: {'yes' if st['token'] else 'NO'}")
    print(f"machine_label: {cfg.get('machine_label')}")
    print(f"hostname: {socket.gethostname()}")
    print(f"timezone: {cfg.get('timezone')}")
    print(f"root_folder_id: {cfg.get('folder_ids', {}).get('root')}")
    print(f"index_sheet_id: {cfg.get('index_sheet_id')}")
    return 0


def _cmd_init(docs, drive, sheets, args) -> int:
    cfg = load_config()
    if args.machine_label:
        cfg["machine_label"] = args.machine_label
        save_config(cfg)
    print("Creating Drive tree and templates…")
    cfg = drive_ops.init_notebook_tree(docs, drive, sheets)
    print(f"Saved config → {CONFIG_PATH}")
    print(f"Root folder id: {cfg['folder_ids']['root']}")
    print(f"Index sheet id: {cfg['index_sheet_id']}")
    print(f"Daily folder id: {cfg['folder_ids']['daily']}")
    return 0


def _cmd_ensure_today(docs, drive, args) -> int:
    cfg = load_config()
    date_str = _date_or_today(cfg, args.date)
    meta = drive_ops.ensure_today_doc(drive, docs, cfg, date_str)
    url = meta.get("webViewLink") or drive_ops.doc_url(meta["id"])
    print(url)
    return 0


def _cmd_link(docs, drive, args) -> int:
    return _cmd_ensure_today(docs, drive, args)


def _cmd_append(docs, drive, sheets, args) -> int:
    cfg = load_config()
    body = Path(args.body_file).expanduser().read_text(encoding="utf-8")
    result = append_entry(
        docs,
        drive,
        sheets,
        cfg,
        title=args.title,
        body_md=body,
        project=args.project,
        workspace=args.workspace,
        repo=args.repo,
        commit=args.commit,
        transcript_id=args.transcript_id,
        figure_paths=args.figure,
        source=args.source,
        date_str=args.date,
    )
    print(result["doc_url"])
    print(f"Appended: {result['title']} ({result['date']} {result['time']})")
    return 0


def _cmd_reconcile(docs, drive, sheets, args) -> int:
    cfg = load_config()
    if args.projects:
        cfg["project_allowlist"] = [
            p.strip() for p in args.projects.split(",") if p.strip()
        ]
    date_str = _date_or_today(cfg, args.date)
    report = build_reconcile_report(
        docs,
        drive,
        sheets,
        cfg,
        date_str,
        skip_github=args.skip_github,
        skip_transcripts=args.skip_transcripts,
    )
    print(report.to_text())

    if args.dry_run or not args.append_missing:
        if not args.append_missing:
            print(
                "Dry report only. Re-run with --append-missing to backfill.",
            )
        return 0

    result = apply_reconcile(
        docs,
        drive,
        sheets,
        cfg,
        report,
        append_transcripts=not args.skip_transcripts,
        append_github=not args.skip_github,
    )
    print(
        f"Appended transcripts={result['appended_transcripts']} "
        f"commits={result['appended_commits']}"
    )
    print(result["doc_url"])
    return 0


def _cmd_github_day(args) -> int:
    cfg = load_config()
    date_str = _date_or_today(cfg, args.date)
    commits = fetch_commits_for_day(cfg, date_str)
    print(format_github_rollup(date_str, commits))
    print(f"Total: {len(commits)} commit(s)")
    return 0


def _cmd_reformat(docs, drive, sheets, args) -> int:
    cfg = load_config()
    date_str = _date_or_today(cfg, args.date)
    print(f"Reformatting daily Doc for {date_str}…")
    result = reformat_day_doc(docs, drive, sheets, cfg, date_str)
    print(result["doc_url"])
    print(
        f"Rewrote {result['n_blocks']} blocks "
        f"({result['n_requests']} Docs API requests)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
