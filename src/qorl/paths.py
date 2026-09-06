"""Shared filesystem locations for commands running from a source checkout."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
