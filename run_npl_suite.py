#!/usr/bin/env python3
"""Root launcher for the Neural Processing Letters reproducibility suite."""

import sys
from pathlib import Path


if __name__ == "__main__":
    experiments = Path(__file__).resolve().parent / "experiments"
    sys.path.insert(0, str(experiments))
    from run_journal_suite import main

    main()
