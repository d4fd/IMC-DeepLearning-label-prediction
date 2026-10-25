# Publication boundary

This repository intentionally contains only reviewed source, documentation,
dependency declarations and synthetic-only checks, with fresh Git history.

Excluded: original images, clinical annotations, source panel/marker tables,
patient/sample identifiers (including pseudonyms), split assignments, fitted
normalization values, image embeddings, predictions, model weights, run logs,
results, figures, reports, notebooks, private plans, archives, credentials and
the private project's Git objects. Schema names and scientific hyperparameters
are code/documentation, not patient records. No private artifact hashes are
published. Upstream public revision/weight identifiers are attributed separately.

## Guardrails

`.gitignore` defaults to excluding everything except reviewed source/documentation
locations. `PUBLIC_FILES.txt` lists the exact approved payload. The standard-library
verifier checks that inventory against all non-ignored files; optionally it checks
the **actual index bytes** and **every reachable Git commit**, not just HEAD.
It rejects unexpected paths/types, binary content, symlinks, oversized files,
syntax errors, known private identifier/path patterns and common secret formats.

```bash
python verify_publication.py
python verify_publication.py --staged
python verify_publication.py --history
git status --short
git diff --cached
git log --all --oneline
```

The initial publication is manually curated as well as machine-checked. Scanner
success is **not proof of absence of arbitrary sensitive text**, encoded data or
unknown secrets. Do not weaken the allowlist or use `git add -f` to include real
artifacts. Any inventory change needs a fresh content review. Even an innocent
CSV, example identifier, notebook attachment or screenshot can disclose data.

## Before a future push

Review exact filenames, text, diffs, staged bytes and all commits being pushed.
Keep real execution inputs/output outside the public payload. Generated patient
hashes and trained heads are also private; lack of pixels does not make them
safe. MIT licensing applies to code, not data. Never copy the original submission
archive into this repository or attach it to a public release/issue.

If sensitive content is ever committed, deleting it in a later commit is not
sufficient: stop publication, assess exposure, rotate exposed credentials and
coordinate cleanup of history and remote copies before proceeding.
