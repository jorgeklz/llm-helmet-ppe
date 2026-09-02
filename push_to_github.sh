#!/usr/bin/env bash
# Publish ONLY the whitelisted code and data to GitHub, and REMOVE any previously
# uploaded files (e.g. an earlier push that included the manuscript). This starts
# a fresh, single-commit history and force-pushes it, so the repository ends up
# containing exactly the files allowed by .gitignore and nothing else.
#
# Repository: https://github.com/jorgeklz/llm-helmet-ppe
# Run:   bash push_to_github.sh
# You will be asked for your GitHub username and a Personal Access Token (PAT)
# as the password when pushing over HTTPS.
#
REPO_URL="https://github.com/jorgeklz/llm-helmet-ppe.git"
cd "$(dirname "$0")"

echo "This will REPLACE the entire contents of $REPO_URL with the current"
echo "whitelisted files (dataset, scripts, logs and the revision package)."
read -r -p "Continue? [y/N] " ok
[ "$ok" = "y" ] || [ "$ok" = "Y" ] || { echo "Aborted."; exit 0; }

# fresh history so nothing old survives
rm -rf .git
git init -q
git branch -M main

# stage only what .gitignore allows
git add -A
git -c user.name="Jorge Parraga-Alava" -c user.email="jorge.parraga@utm.edu.ec" \
    commit -q -m "Dataset, code and per-run logs for the LLM-helmet PPE study"

echo
echo "Files that will be published (top level):"
git ls-files | sed 's#/.*##' | sort -u
echo "Total files: $(git ls-files | wc -l)"
echo

git remote add origin "$REPO_URL"
echo "Force-pushing to $REPO_URL (replaces remote main)..."
echo "If prompted, enter your GitHub username and a Personal Access Token as the password."
git push -u origin main --force
echo "Done. Verify at https://github.com/jorgeklz/llm-helmet-ppe"
