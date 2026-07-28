## Agent skills

### Issue tracker

Issues live in GitHub Issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

**Important:** `gh` requires a GitHub App token that is not available as a persistent env var in agent shell sessions. Always prefix `gh` commands with:

```
GH_TOKEN=$(bash ~/.config/gh-app/gh-app-token.sh 2>/dev/null) gh ...
```

This mirrors the `gh-app` shell function defined in `~/.bashrc`.

### Triage labels

Default five-label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
