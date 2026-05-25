from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from aihub.phase8_vpcd import cli

    return cli(argv or sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
