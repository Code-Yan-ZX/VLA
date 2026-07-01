#!/usr/bin/env python
"""P0 smoke-test job for the queue mechanism.
Does NOT depend on torch/GPU (so it passes before vtc is built).
Writes a marker into runs/ and exits 0. Replace with real jobs in P2/P3.
"""
import os, sys, time, platform

RUNS = os.path.join(os.path.dirname(__file__), os.pardir, "runs")
os.makedirs(RUNS, exist_ok=True)
marker = os.path.join(RUNS, "_dummy_marker.txt")

print(f"[dummy] python  : {sys.version.split()[0]} ({sys.executable})")
print(f"[dummy] platform: {platform.platform()}")
print(f"[dummy] cwd     : {os.getcwd()}")
time.sleep(1)
with open(marker, "w") as f:
    f.write(f"smoke ok @ {time.time()}\n")
print(f"[dummy] wrote {marker}")
print("[dummy] OK")
sys.exit(0)
