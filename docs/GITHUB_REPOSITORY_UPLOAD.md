# GitHub Repository Upload Guide

This repository is fully committed locally. Use this guide to attach all project files to a GitHub repository.

## Prerequisites

- A GitHub repository created in your GitHub account or organization.
- Git installed locally.
- Authentication configured with SSH keys or a GitHub personal access token.

## Option 1: Existing GitHub Repository

Replace the placeholder URL with your repository URL:

```bash
git remote add origin https://github.com/<owner>/<repository>.git
git branch -M main
git push -u origin main
```

If the `origin` remote already exists, update it instead:

```bash
git remote set-url origin https://github.com/<owner>/<repository>.git
git push -u origin main
```

## Option 2: GitHub CLI

If the GitHub CLI is installed and authenticated:

```bash
gh repo create <owner>/<repository> --private --source=. --remote=origin --push
```

Use `--public` instead of `--private` if the project should be publicly visible.

## Verify the Upload

After pushing, verify that the repository contains these top-level project assets:

- `README.md`
- `requirements.txt`
- `requirements-ml.txt`
- `Dockerfile`
- `Procfile`
- `runtime.txt`
- `.github/workflows/ci.yml`
- `src/`
- `streamlit_app/`
- `notebooks/`
- `tests/`
- `reports/`
- `models/`
- `data/feature_engineered_dataset/`

Run:

```bash
git status --short --branch
git remote -v
git ls-files | sort
```

The working tree should be clean before and after pushing. The `git ls-files` output lists every file that will be uploaded to GitHub.

## Notes on Large or Sensitive Files

- Do not commit raw private customer data, secrets, API keys, or credentials.
- Large trained model artifacts should usually be stored in GitHub Releases, object storage, or an artifact registry rather than normal Git history.
- The included `.gitignore` intentionally ignores generated model binaries, report outputs, caches, and local environment files.
