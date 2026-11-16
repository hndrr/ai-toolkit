# Local Git safety hooks

This checkout is configured with `core.hooksPath=.githooks`.

- `.githooks/pre-push` blocks direct pushes to `ostris/ai-toolkit`.
- The local `upstream` remote is fetch-only; use `git fetch upstream` to read upstream changes.
- `.githooks/post-push` prints a compare URL whose base is `hndrr/ai-toolkit`.
- Use `scripts/create_pr_to_fork.sh` when a pull request should stay inside `hndrr/ai-toolkit`.

If this repository is cloned again, enable the hooks with:

```bash
git config core.hooksPath .githooks
```
