"""Compatibility shim for paper Section 5 (label audit).

The audit is implemented end-to-end in ``cleanlab_label_audit.py`` (cleanlab-based
93/100 flag analysis) and ``near_dup_audit.py`` (pHash near-duplicate analysis).
This shim is the single entry point referenced by the README quick-start; it
runs both audits in sequence and writes a consolidated report to
``results/label_audit_report.md``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, cwd=REPO).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cleanlab", action="store_true")
    ap.add_argument("--skip-phash", action="store_true")
    args = ap.parse_args()

    rc = 0
    if not args.skip_cleanlab:
        rc |= run([sys.executable, "scripts/cleanlab_label_audit.py"])
    if not args.skip_phash:
        rc |= run([sys.executable, "scripts/near_dup_audit.py"])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
