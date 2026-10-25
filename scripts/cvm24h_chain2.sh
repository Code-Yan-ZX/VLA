#!/bin/bash
set -u
cd "$(git rev-parse --show-toplevel)"
bash scripts/cvm24h_run_timing.sh
bash scripts/cvm24h_run_q10.sh
