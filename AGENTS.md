# positronic-engram — public contributor guide

Public engine library: `engine/src/memeng` (`import memeng`).

Plugin brain access: `positronic-opencode-plugin` exposes `positronic.*` tools + `/positronic:*` slashes globally. Direct `memeng` hacking: `engine/src/memeng` + `positronic-opencode-plugin/src/brains.py`.

```bash
pytest engine/tests/
ruff check engine/src/memeng/ tests/
```

No PII. Henry's deployment lives in `../positronic-private` (private, gitignored).
