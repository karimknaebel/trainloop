test:
    uv run pytest

[confirm]
bump bump="minor":
    test -z "$(git status --porcelain)" || (echo "error: repo must be clean before bump" >&2; exit 1)
    uv version --bump {{ bump }}
    git commit -m "pkg: bump version to $(uv version --short)" pyproject.toml uv.lock
    @echo "created bump commit for $(uv version --short). next: git push. then run just release."

[confirm]
release:
    gh release create v$(uv version --short) --generate-notes
