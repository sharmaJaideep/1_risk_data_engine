from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk Data Engine CLI")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(f"Risk Data Engine initialized with data directory: {args.data_dir}")


if __name__ == "__main__":
    main()
