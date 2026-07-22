"""Fetch today's GitHub commits for configured owners via ``gh``."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo


@dataclass
class GithubCommit:
    """One commit on a watched GitHub owner/repo.

    Attributes
    ----------
    owner : str
        GitHub user or org.
    repo : str
        Repository name.
    sha : str
        Full commit SHA.
    message : str
        Commit subject (first line).
    url : str
        HTML URL.
    author_date : str
        ISO author date.
    author_login : str
        Author login if known.
    """

    owner: str
    repo: str
    sha: str
    message: str
    url: str
    author_date: str
    author_login: str = ""

    @property
    def short_sha(self) -> str:
        return self.sha[:7]

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def fetch_commits_for_day(
    cfg: dict[str, Any], date_str: str
) -> list[GithubCommit]:
    """Return commits authored on ``date_str`` under configured owners.

    Parameters
    ----------
    cfg : dict
        Configuration with ``github`` section and ``timezone``.
    date_str : str
        ``YYYY-MM-DD``.

    Returns
    -------
    list of GithubCommit
        Deduplicated by SHA, sorted by author date.
    """

    gh_cfg = cfg.get("github") or {}
    owners = gh_cfg.get("owners") or ["jluastro", "MovingUniverseLab"]
    authors = list(gh_cfg.get("authors") or [])
    # Auto-detect the logged-in gh user and always include them.
    gh_login = _gh_login()
    if gh_login and gh_login not in authors:
        authors.append(gh_login)
    if not authors:
        authors = ["jluastro"]
    block = set(gh_cfg.get("repo_blocklist") or [])
    active = gh_cfg.get("active_repos") or []
    tz = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))

    day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    day_end = day_start + timedelta(days=1)
    since = day_start.isoformat()
    until = day_end.isoformat()

    commits: dict[str, GithubCommit] = {}

    # Prefer GitHub commit search per author+owner.
    for author in authors:
        for owner in owners:
            for c in _search_commits(author, owner, date_str):
                if c.repo in block or f"{c.owner}/{c.repo}" in block:
                    continue
                if not _date_in_range(c.author_date, day_start, day_end):
                    continue
                commits[c.sha] = c

    # Fallback / supplement: active repos list via commits API.
    for full in active:
        if "/" not in full:
            continue
        owner, repo = full.split("/", 1)
        if repo in block or full in block:
            continue
        for c in _list_repo_commits(owner, repo, since, until, authors):
            commits[c.sha] = c

    # If search returned nothing, try listing repos for each owner (capped).
    if not commits:
        for owner in owners:
            for repo in _list_owner_repos(owner, limit=30):
                if repo in block or f"{owner}/{repo}" in block:
                    continue
                for c in _list_repo_commits(owner, repo, since, until, authors):
                    commits[c.sha] = c

    out = list(commits.values())
    out.sort(key=lambda c: c.author_date)
    return out


def _gh_login() -> str:
    """Return the authenticated GitHub login, or empty string."""

    proc = _run_gh(["api", "user", "-q", ".login"])
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()



def format_github_rollup(date_str: str, commits: list[GithubCommit]) -> str:
    """Format a Markdown body for the GitHub commits section.

    Parameters
    ----------
    date_str : str
        Calendar date.
    commits : list of GithubCommit
        Commits to include.

    Returns
    -------
    str
        Markdown body.
    """

    lines = [f"GitHub commits for {date_str}.\n"]
    by_repo: dict[str, list[GithubCommit]] = {}
    for c in commits:
        by_repo.setdefault(c.full_name, []).append(c)

    for repo in sorted(by_repo):
        lines.append(f"### {repo}")
        for c in by_repo[repo]:
            subj = c.message.split("\n", 1)[0][:120]
            lines.append(f"- {c.short_sha} — {subj} ({c.url})")
        lines.append("")
    return "\n".join(lines)


def _run_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` and return the completed process."""

    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _search_commits(
    author: str, owner: str, date_str: str
) -> list[GithubCommit]:
    """Use ``gh search commits`` for author-date on an owner."""

    # Also try user: for personal accounts named like jluastro.
    # org: fails for user namespaces; user: fails for orgs — try both.
    queries = [
        f"author:{author} author-date:{date_str} org:{owner}",
        f"author:{author} author-date:{date_str} user:{owner}",
        f"author-date:{date_str} org:{owner}",
        f"author-date:{date_str} user:{owner}",
    ]
    found: list[GithubCommit] = []
    seen: set[str] = set()
    for q in queries:
        proc = _run_gh(
            [
                "search",
                "commits",
                q,
                "--json",
                "sha,commit,repository,url,author",
                "--limit",
                "100",
            ]
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            items = json.loads(proc.stdout)
        except json.JSONDecodeError:
            continue
        for item in items:
            repo_info = item.get("repository") or {}
            full = repo_info.get("nameWithOwner") or repo_info.get("name") or ""
            if "/" in full:
                o, r = full.split("/", 1)
            else:
                o, r = owner, full
            commit = item.get("commit") or {}
            author_info = commit.get("author") or {}
            msg = (commit.get("message") or "").strip()
            sha = item.get("sha") or ""
            if not sha or sha in seen:
                continue
            # Prefer canonical commit HTML URL.
            url = f"https://github.com/{o}/{r}/commit/{sha}" if o and r else (
                item.get("url") or ""
            )
            login = ""
            a = item.get("author")
            if isinstance(a, dict):
                login = a.get("login") or ""
            # When query omitted author:, keep only configured authors if login known.
            if author and login and login.lower() != author.lower():
                # Still allow if this query was author-scoped.
                if q.startswith("author-date:"):
                    continue
            seen.add(sha)
            found.append(
                GithubCommit(
                    owner=o,
                    repo=r,
                    sha=sha,
                    message=msg,
                    url=url,
                    author_date=author_info.get("date") or "",
                    author_login=login or author,
                )
            )
    return found


def _list_repo_commits(
    owner: str,
    repo: str,
    since: str,
    until: str,
    authors: list[str],
) -> list[GithubCommit]:
    """List commits on a repo in a time window, filter by author login."""

    proc = _run_gh(
        [
            "api",
            f"repos/{owner}/{repo}/commits",
            "--method",
            "GET",
            "-f",
            f"since={since}",
            "-f",
            f"until={until}",
            "--paginate",
        ]
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []

    try:
        # gh --paginate may concatenate JSON arrays; handle both.
        raw = proc.stdout.strip()
        if raw.startswith("["):
            # Possibly multiple arrays concatenated.
            items: list[Any] = []
            decoder = json.JSONDecoder()
            idx = 0
            while idx < len(raw):
                while idx < len(raw) and raw[idx].isspace():
                    idx += 1
                if idx >= len(raw):
                    break
                chunk, offset = decoder.raw_decode(raw[idx:])
                idx += offset
                if isinstance(chunk, list):
                    items.extend(chunk)
                elif isinstance(chunk, dict):
                    items.append(chunk)
        else:
            items = []
    except json.JSONDecodeError:
        return []

    authors_l = {a.lower() for a in authors}
    out: list[GithubCommit] = []
    for item in items:
        sha = item.get("sha") or ""
        commit = item.get("commit") or {}
        msg = (commit.get("message") or "").strip()
        author_info = commit.get("author") or {}
        date = author_info.get("date") or ""
        login = ""
        if isinstance(item.get("author"), dict):
            login = item["author"].get("login") or ""
        # Filter: match login if present, else match author name loosely.
        if login and login.lower() not in authors_l:
            # Also allow commits where git name matches author list.
            name = (author_info.get("name") or "").lower()
            if not any(a.lower() in name for a in authors):
                continue
        url = item.get("html_url") or f"https://github.com/{owner}/{repo}/commit/{sha}"
        out.append(
            GithubCommit(
                owner=owner,
                repo=repo,
                sha=sha,
                message=msg,
                url=url,
                author_date=date,
                author_login=login,
            )
        )
    return out


def _list_owner_repos(owner: str, limit: int = 30) -> list[str]:
    """List repository names for a user or org."""

    proc = _run_gh(
        [
            "repo",
            "list",
            owner,
            "--limit",
            str(limit),
            "--json",
            "name",
        ]
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        items = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    return [i.get("name", "") for i in items if i.get("name")]


def _date_in_range(
    iso_date: str, start: datetime, end: datetime
) -> bool:
    """Return True if ISO date falls in [start, end)."""

    if not iso_date:
        return True
    try:
        from dateutil import parser as date_parser

        dt = date_parser.isoparse(iso_date)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=start.tzinfo)
        return start <= dt.astimezone(start.tzinfo) < end
    except Exception:
        return True
