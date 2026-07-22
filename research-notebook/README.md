# research-notebook

Append research notes to **native Google Docs** from any Cursor window on any
machine. End-of-day reconcile scans local Cursor agent transcripts and GitHub
commits on `jluastro` / `MovingUniverseLab` repos.

## Install (laptop or Berkeley host)

```bash
zsh -lic 'cd ~/code/python/research-notebook && pip install -e .'
# or in astro:
zsh -lic 'cd ~/code/python/research-notebook && mamba run -n astro pip install -e .'
```

Confirm:

```bash
zsh -lic 'which research-notebook && research-notebook status'
```

## Google OAuth (once)

1. Create a Google Cloud project; enable **Docs**, **Drive**, and **Sheets** APIs.
2. Create an OAuth **Desktop** client; download JSON to:

   `~/.config/research_notebook/client_secret.json`

3. Run:

```bash
research-notebook setup-oauth-docs   # writes SETUP_OAUTH.md
research-notebook auth               # browser consent → token.json
research-notebook init               # creates Drive tree + Index
```

Copy `~/.config/research_notebook/{client_secret.json,token.json,config.yaml}`
to other machines (e.g. dragon) instead of re-auth on headless hosts.

## Cursor integration

Personal skill + always-on user rule (installed by `scripts/install_cursor_hooks.sh`):

- `add these results to my research notebook`
- `add a summary of this task to my research notebook`
- `reconcile my research notebook for today`

## CLI

| Command | Purpose |
|---------|---------|
| `research-notebook auth` | OAuth |
| `research-notebook init` | Drive folders, templates, Index Sheet |
| `research-notebook ensure-today` / `link` | Today's Doc URL |
| `research-notebook append --title … --body-file … [--figure …]` | Append entry |
| `research-notebook reconcile-day [--dry-run\|--append-missing]` | Transcripts + GitHub |
| `research-notebook github-day` | List today's commits |
| `research-notebook status` | Config / auth check |

### Append example

```bash
cat > /tmp/rn_body.md <<'EOF'
### What I did
Tested GP residual weighting.

### Results
Residuals look consistent with white noise.

### Reproduce
```bash
cd /path/to/bagle && PYTHONPATH=src python …
```
EOF

research-notebook append \
  --title "bagle GP residuals" \
  --project bagle \
  --workspace "$PWD" \
  --repo bagle \
  --commit "$(git rev-parse HEAD)" \
  --body-file /tmp/rn_body.md \
  --figure ./plot.png
```

## Config

`~/.config/research_notebook/config.yaml` — folder IDs, timezone, GitHub owners,
transcript allow/block lists. See defaults in `research_notebook.config`.

## Sharing with students

Copy a daily Doc (or paste one entry) into Drive folder `07_Shared/`, then share
**that copy only**.
