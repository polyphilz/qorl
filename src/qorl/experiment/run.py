"""Shared execution entrypoint; stage implementation is gated on 036 Phase 3."""

import sys
from pathlib import Path


def main(experiment_directory: Path) -> int:
    """Fail explicitly instead of claiming that an unimplemented stage ran."""
    print(
        f"qorl: execution is not implemented (036 Phase 3): {experiment_directory}",
        file=sys.stderr,
    )
    return 1
