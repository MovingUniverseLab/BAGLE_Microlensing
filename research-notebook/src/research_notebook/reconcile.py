"""End-of-day reconcile: transcripts + GitHub → gap report / backfill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from research_notebook import drive_ops
from research_notebook.github_commits import (
    GithubCommit,
    fetch_commits_for_day,
    format_github_rollup,
)
from research_notebook.notebook import (
    append_entry,
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
    """Backfill missing transcripts and GitHub rollup from a report.

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

    appended_t = 0
    doc_url = ""

    if append_transcripts:
        for t in report.missing_transcripts:
            body = (
                f"### What I did\n"
                f"Cursor session in project `{t.project}`.\n\n"
                f"**First query:** {t.first_query}\n\n"
            )
            if t.sample_assistant:
                body += f"### Notes from assistant\n{t.sample_assistant}\n\n"
            body += (
                f"### Reproduce\n"
                f"See transcript: `{t.path}`\n"
            )
            result = append_entry(
                docs,
                drive,
                sheets,
                cfg,
                title=f"{t.project}: {t.first_query[:60]}",
                body_md=body,
                project=t.project,
                workspace=t.path,
                transcript_id=t.transcript_id,
                source="reconcile",
                date_str=report.date,
            )
            doc_url = result["doc_url"]
            appended_t += 1

    appended_c = 0
    if append_github and report.new_commits:
        body = format_github_rollup(report.date, report.new_commits)
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
            commit=report.new_commits[0].sha,
            extra_meta=[
                f"commits: {len(report.new_commits)} new",
            ],
        )
        doc_url = result["doc_url"]
        # Index one row per commit for SHA idempotency.
        from research_notebook.notebook import append_index_row

        machine = cfg.get("machine_label", "")
        for c in report.new_commits[1:]:
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
            appended_c += 1
        appended_c += 1  # first commit counted via append_entry

    # End-of-day summary block.
    summary_body = (
        f"### End-of-day reconcile\n"
        f"- Transcripts missing→appended: {appended_t} "
        f"(covered {len(report.covered_transcripts)}, "
        f"skipped {len(report.skipped_transcripts)})\n"
        f"- GitHub new commits logged: {len(report.new_commits)}\n"
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
        "appended_commits": len(report.new_commits) if append_github else 0,
        "doc_url": doc_url,
    }
