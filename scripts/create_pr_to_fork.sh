#!/bin/sh

# Create a PR inside the user's fork, never against the upstream repository.
set -eu

expected_repo="hndrr/ai-toolkit"
origin_url="$(git remote get-url origin 2>/dev/null || true)"

repo_path="$(printf '%s\n' "$origin_url" \
  | sed -E \
      -e 's#^[^:/]+://[^/]+/##' \
      -e 's#^[^@]+@[^:]+:##' \
      -e 's#\.git/?$##' \
      -e 's#/$##' \
  | tr '[:upper:]' '[:lower:]')"

if [ "$repo_path" != "$expected_repo" ]; then
  echo "Refusing to create a PR: origin must be $expected_repo, got $repo_path." >&2
  exit 1
fi

for arg in "$@"; do
  case "$arg" in
    --repo|--repo=*)
      echo "Do not override --repo; this command is locked to $expected_repo." >&2
      exit 1
      ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required. No PR was created." >&2
  exit 1
fi

exec gh pr create --repo "$expected_repo" --base main "$@"
