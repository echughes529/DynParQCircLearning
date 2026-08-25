"""Inline the figures into the artifact page so it is self-contained.

Run:
    python -m src.diagnostics.build_artifact

A published artifact is served under a strict content-security policy that blocks
every external host, and it is a single file with no sibling directory to serve
`figures/*.png` from. So the page has to carry its images inside it: each
`src="figures/x.png"` is replaced with a base64 `data:` URI of that file.

Reads docs/traj_vs_purified_artifact.html and writes the built page alongside it,
leaving the source template editable with normal relative paths.
"""

import base64
import os
import re
import sys

SRC = "docs/traj_vs_purified_artifact.html"
OUT = "docs/traj_vs_purified_artifact.built.html"
FIGDIR = "docs"

# 16 MB is the hard limit on a rendered artifact, and base64 inflates by 4/3.
SIZE_LIMIT = 16 * 1024 * 1024


def main():
    with open(SRC) as fh:
        html = fh.read()

    missing = []

    def inline(match):
        rel = match.group(1)
        path = os.path.join(FIGDIR, rel)
        if not os.path.exists(path):
            missing.append(rel)
            return match.group(0)
        with open(path, "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
        print(f"  inlined {rel}  ({len(payload) / 1024:.0f} KB encoded)")
        return f'src="data:image/png;base64,{payload}"'

    built = re.sub(r'src="(figures/[^"]+\.png)"', inline, html)

    if missing:
        print(f"ERROR: referenced figures do not exist: {', '.join(missing)}", file=sys.stderr)
        print("Run: python -m src.diagnostics.plot_traj_vs_purified", file=sys.stderr)
        return 1

    if len(built.encode()) > SIZE_LIMIT:
        print(f"ERROR: built page is {len(built) / 1e6:.1f} MB, over the 16 MB artifact limit.",
              file=sys.stderr)
        return 1

    with open(OUT, "w") as fh:
        fh.write(built)
    print(f"\nWrote {OUT}  ({len(built.encode()) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
