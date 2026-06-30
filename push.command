#!/bin/bash
# Double-click this file in Finder to push all changes to GitHub.
# It stages everything, commits with today's date, and pushes to main.

cd "$(dirname "$0")" || exit 1

echo "Cleaning up any stale git locks..."
rm -f .git/HEAD.lock .git/index.lock

echo "Staging all changes..."
git add -A

if git diff --cached --quiet; then
    echo "Nothing new to commit — pushing any unpushed commits instead."
else
    git commit -m "Update article code ($(date +%Y-%m-%d))"
fi

echo "Pushing to GitHub..."
git push origin main

echo ""
echo "Done. You can close this window."
