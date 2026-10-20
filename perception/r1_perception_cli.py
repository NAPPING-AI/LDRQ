#!/usr/bin/env python3
"""Entry point; see r1_perception/cli.py."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r1_perception.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
