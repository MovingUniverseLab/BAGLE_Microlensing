"""End-of-day reconcile: transcripts + GitHub → gap report / backfill."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from dateutil import parser as date_parser

from research_notebook.github_commits import (
    GithubCommit,
    fetch_commits_for_day,
    format_github_rollup,
)
from research_notebook.notebook import (
    append_entry,
    append_index_row,
    index_coverage_for_date,
    today_doc_text,
)
from research_notebook.transcripts import TranscriptSummary, scan_transcripts_for_day


@dataclass
class ReconcileReport:
    """Gap report for one calendar day.

    Attributes
    ----------
    date : str
        ``YYYY-MM-DD``.
    covered_transcripts : list
        Already indexed transcripts.
    missing_transcripts : list
        Research-like transcripts not yet logged.
    skipped_transcripts : list
        Blocked / empty transcripts.
    new_commits : list
        GitHub commits not yet in Index.
    known_commits : list
        Commits already indexed.
    """

    date: str
    covered_transcripts: list[TranscriptSummary] = field(default_factory=list)
    missing_transcripts: list[TranscriptSummary] = field(default_factory=list)
    skipped_transcripts: list[TranscriptSummary] = field(default_factory=list)
    new_commits: list[GithubCommit] = field(default_factory=list)
    known_commits: list[GithubCommit] = field(default_factory=list)

    def to_text(self) -> str:
        """Human-readable checklist.

        Returns
        -------
        str
            Report text.
        """

        lines = [f"Reconcile report for {self.date}", ""]
        lines.append(
            f"Transcripts: {len(self.missing_transcripts)} missing, "
            f"{len(self.covered_transcripts)} covered, "
            f"{len(self.skipped_transcripts)} skipped"
        )
        for t in self.missing_transcripts:
            q = t.first_query.replace("\n", " ")[:100]
            lines.append(f"  [MISSING] {t.transcript_id[:8]}…  {t.project}: {q}")
        for t in self.covered_transcripts:
            lines.append(f"  [ok]      {t.transcript_id[:8]}…  {t.project}")
        for t in self.skipped_transcripts:
            lines.append(
                f"  [skip]    {t.transcript_id[:8]}…  {t.project} ({t.skip_reason})"
            )

        lines.append("")
        lines.append(
            f"GitHub: {len(self.new_commits)} new, {len(self.known_commits)} known"
        )
        for c in self.new_commits:
            subj = c.message.split("\n", 1)[0][:80]
            lines.append(f"  [NEW] {c.full_name}@{c.short_sha} — {subj}")
        for c in self.known_commits:
            lines.append(f"  [ok]  {c.full_name}@{c.short_sha}")
        lines.append("")
        return "\n".join(lines)


def build_reconcile_report(
    docs: Any,
    drive: Any,
    sheets: Any,
    cfg: dict[str, Any],
    date_str: str,
    *,
    skip_github: bool = False,
    skip_transcripts: bool = False,
) -> ReconcileReport:
    """Scan transcripts and GitHub; compare to Index coverage.

    Parameters
    ----------
    docs, drive, sheets :
        Google API clients (docs/drive used for optional Doc text scan).
    cfg : dict
        Configuration.
    date_str : str
        Calendar date.
    skip_github : bool, optional
        Skip GitHub pass.
    skip_transcripts : bool, optional
        Skip transcript pass.

    Returns
    -------
    ReconcileReport
        Gap report.
    """

    coverage = index_coverage_for_date(sheets, cfg, date_str)
    doc_text = ""
    try:
        doc_text = today_doc_text(docs, drive, cfg, date_str)
    except Exception:
        doc_text = ""

    report = ReconcileReport(date=date_str)

    if not skip_transcripts:
        for t in scan_transcripts_for_day(cfg, date_str):
            if not t.research_like:
                report.skipped_transcripts.append(t)
                continue
            if (
                t.transcript_id in coverage["transcript_ids"]
                or (t.transcript_id and t.transcript_id in doc_text)
            ):
                report.covered_transcripts.append(t)
            else:
                report.missing_transcripts.append(t)

    if not skip_github:
        commits = fetch_commits_for_day(cfg, date_str)
        for c in commits:
            if c.sha in coverage["commits"] or c.sha[:7] in doc_text:
                report.known_commits.append(c)
            else:
                report.new_commits.append(c)

    return report


def apply_reconcile(
    docs: Any,
    drive: Any,
    sheets: Any,
    cfg: dict[str, Any],
    report: ReconcileReport,
    *,
    append_transcripts: bool = True,
    append_github: bool = True,
) -> dict[str, Any]:
    """Backfill missing transcripts and GitHub rollup in time order.

    Parameters
    ----------
    docs, drive, sheets :
        API clients.
    cfg : dict
        Configuration.
    report : ReconcileReport
        Previously built report.
    append_transcripts : bool, optional
        Backfill missing transcript entries.
    append_github : bool, optional
        Append GitHub rollup for new commits.

    Returns
    -------
    dict
        Counts and final doc URL.
    """

    from research_notebook.reformat import short_project_name, short_title_from_query

    appended_t = 0
    appended_c = 0
    doc_url = ""

    tz = ZoneInfo(cfg.get("timezone", "America/Los_Angeles"))
    day_start = datetime.strptime(report.date, "%Y-%m-%d").replace(tzinfo=tz)

    # Timed queue keeps appends chronological across transcripts + GitHub.
    queue: list[tuple[datetime, str, Any]] = []
    if append_transcripts:
        for t in report.missing_transcripts:
            when = t.start_time or day_start
            queue.append((when, "transcript", t))
    if append_github and report.new_commits:
        times = []
        for c in report.new_commits:
            if c.author_date:
                try:
                    dt = date_parser.isoparse(c.author_date)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=tz)
                    times.append(dt.astimezone(tz))
                except Exception:
                    pass
        when = min(times) if times else day_start
        queue.append((when, "github", report.new_commits))

    queue.sort(key=lambda item: item[0])

    for _when, kind, payload in queue:
        if kind == "transcript":
            t = payload
            proj = short_project_name(t.project)
            query = re.sub(r"\s+", " ", (t.first_query or "").strip())
            if len(query) > 500:
                query = query[:499] + "…"
            body = (
                f"### Summary\n"
                f"Cursor session in `{proj}`.\n\n"
                f"### First query\n"
                f"{query}\n\n"
            )
            if t.sample_assistant:
                note = re.sub(r"\s+", " ", t.sample_assistant.strip())
                if len(note) > 400:
                    note = note[:399] + "…"
                body += f"### Notes\n{note}\n\n"
            body += (
                f"### Results\n"
                f"- (Reconcile backfill — add tests/timings/plots if known.)\n\n"
                f"### Reproduce\n"
                f"- transcript-id: `{t.transcript_id}`\n"
                f"- path: `{t.path}`\n"
            )
            result = append_entry(
                docs,
                drive,
                sheets,
                cfg,
                title=f"{proj}: {short_title_from_query(t.first_query)}",
                body_md=body,
                project=proj,
                workspace=t.path,
                transcript_id=t.transcript_id,
                source="reconcile",
                date_str=report.date,
            )
            doc_url = result["doc_url"]
            appended_t += 1
        elif kind == "github":
            commits = payload
            body = format_github_rollup(report.date, commits)
            result = append_entry(
                docs,
                drive,
                sheets,
                cfg,
                title=f"GitHub commits — {report.date}",
                body_md=body,
                project="github",
                source="github",
                date_str=report.date,
                commit=commits[0].sha,
                extra_meta=[f"commits: {len(commits)} new"],
            )
            doc_url = result["doc_url"]
            machine = cfg.get("machine_label", "")
            for c in commits[1:]:
                append_index_row(
                    sheets,
                    cfg,
                    {
                        "date": report.date,
                        "time": "",
                        "title": f"{c.full_name}@{c.short_sha}",
                        "project": c.repo,
                        "doc_url": doc_url,
                        "machine": machine,
                        "workspace": "",
                        "repo": c.full_name,
                        "commit": c.sha,
                        "transcript_id": "",
                        "asset_folder_url": "",
                        "status": "logged",
                        "source": "github",
                    },
                )
            appended_c = len(commits)

    summary_body = (
        f"### End-of-day reconcile\n"
        f"- Transcripts missing→appended: {appended_t} "
        f"(covered {len(report.covered_transcripts)}, "
        f"skipped {len(report.skipped_transcripts)})\n"
        f"- GitHub new commits logged: {appended_c}\n"
    )
    result = append_entry(
        docs,
        drive,
        sheets,
        cfg,
        title="End-of-day reconcile",
        body_md=summary_body,
        project="reconcile",
        source="reconcile",
        date_str=report.date,
        status="reconcile-summary",
    )
    doc_url = result["doc_url"]

    return {
        "appended_transcripts": appended_t,
        "appended_commits": appended_c,
        "doc_url": doc_url,
    }
