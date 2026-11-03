<!-- codebox:board:start -->
## Board

Deferred work for this repo lives on its GitHub Projects board (Triage, Planned, In Progress, Done). Use the `board` command (`board --help`) to keep it current:

1. **Work starts on a card:** move it to In Progress immediately: `board move N in-progress`.
2. **A PR is opened:** items it completes get `Closes #N` in the PR body (GitHub moves them to Done on merge). Items it only partly advances get `Refs #N`, are moved to In Progress, and get a short comment on what changed and what's left: `board move N in-progress --comment - <<'EOF'` (text, then `EOF`).
3. **Non-PR work** (cluster runs, investigations, dataset publishes): when a card's "Done when:" is met, close it with a comment linking the evidence (W&B run, HF tag, commit): `board move N done --comment - <<'EOF'` (text, then `EOF`).

Make these moves without asking, and report them in one line, e.g. `Board: #12 → Done (PR #49), #15 → In Progress`.

New ideas go in Triage (the `codebox-defer` skill does this). Check `board search TERMS` first; on an open match, `board comment N --body - <<'EOF'` instead of filing a duplicate. A card must stand on its own weeks later:

```
board add --title 'Short, specific title' --body - <<'EOF'
**Why:** what prompted it and what problem it solves.
**Context:** relevant files, commits, PRs, runs (W&B/HF links), constraints already known.
**Done when:** one concrete, checkable line.
EOF
```

`--body -` and `--comment -` read the text from stdin; otherwise the text is the flag's value.

Quote titles with single quotes and send bodies/comments through a quoted heredoc so backticks and $ aren't expanded.
<!-- codebox:board:end -->
