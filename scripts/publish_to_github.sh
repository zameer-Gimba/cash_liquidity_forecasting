#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <github-repository-url> [branch]" >&2
  echo "Example: $0 https://github.com/owner/agent-liquidity-prediction.git main" >&2
  exit 2
fi

repo_url="$1"
branch="${2:-main}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree is not clean. Commit or stash changes before publishing." >&2
  git status --short
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$repo_url"
else
  git remote add origin "$repo_url"
fi

git branch -M "$branch"
git push -u origin "$branch"
