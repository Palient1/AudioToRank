from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CliArgs:
    input_dir: Path
    start: str | None
    end: str | None


def build_parser(default_input_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process audio files in a folder and send to LLM agent"
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="Folder with audio files",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start of period (YYYY-MM-DD or ISO datetime)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End of period (YYYY-MM-DD or ISO datetime)",
    )
    return parser


def parse_cli_args(default_input_dir: Path, argv: Iterable[str] | None = None) -> CliArgs:
    parser = build_parser(default_input_dir)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return CliArgs(input_dir=Path(args.input_dir), start=args.start, end=args.end)
