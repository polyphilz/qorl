#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from qorl.resources import load_runtime_profile, validate_host_topology  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one PostgreSQL runtime-profile slot as environment variables."
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--validate-host", action="store_true")
    args = parser.parse_args()

    profile = load_runtime_profile(REPOSITORY_ROOT, args.profile)
    if not 0 <= args.slot < len(profile.workers):
        raise SystemExit(f"runtime profile has no slot {args.slot}")
    resources = profile.workers[args.slot]
    if args.validate_host:
        validate_host_topology((resources,))

    values = {
        **resources.compose_environment,
        "QORL_POSTGRES_RUNTIME_PROFILE": str(profile.path),
        "QORL_POSTGRES_RUNTIME_PROFILE_ID": profile.profile_id,
        "QORL_POSTGRES_RUNTIME_PROFILE_SHA256": profile.sha256,
    }
    for name, value in sorted(values.items()):
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
