#!/usr/bin/env python3

import subprocess
import time

# Configuration
solver_script = "./solver.py"  # path to your solver
points_file = "points_10000_xy.xlsx"
tol = 1e-6
iter = 10
iters_list = [i *1000 for i in range(1,iter+1)]

for iters in iters_list:
    print(f"\nRunning solver with iters={iters}...")

    start = time.time()

    # Run solver.py as a subprocess
    result = subprocess.run(
        ["python3", solver_script, "-p", points_file, "-i", str(iters), "-t", str(tol)],
        capture_output=True,
        text=True
    )

    end = time.time()
    elapsed = end - start

    if result.returncode != 0:
        print(f"Solver failed:\n{result.stderr}")
    else:
        # Extract the score from stdout
        stdout_lines = result.stdout.strip().split("\n")
        score_line = [line for line in stdout_lines if "score:" in line]
        score = score_line[0] if score_line else "N/A"

        print(f"{score}, elapsed time: {elapsed:.4f} seconds")
