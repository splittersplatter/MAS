# AI3403 Multi-Agent Systems - Assignment 3

This repository contains the complete Assignment 3 submission for Dhruv Goyal (EP23BTECH11008), including the final report, reproducible simulation code, figures, and the required name-formation animation.

## Contents

- `Assignment_3_Solution.pdf` - final submission report covering all four problems.
- `Assignment_3_Solution.tex` - report source.
- `Assignment_3_Problem_1.py` - 20-agent formation control for `DHRUV`; creates the GIF and supporting figures.
- `Assignment_3_Problem_2.py` - distributed gradient-tracking simulation for moving-target estimation.
- `Assignment_3_Problem_3.py` - capacitated task-allocation ADMM simulation for three cost matrices.
- `assets/` - generated figures and `p1_DHRUV_formation.gif` referenced by the report.
- `IIT_Hyderabad_Multi_Agent_Systems_Assignment_3.pdf` - original assignment brief.

## Reproduce the numerical results

Use Python 3.10 or later. Create an environment, install the dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python Assignment_3_Problem_1.py
python Assignment_3_Problem_2.py
python Assignment_3_Problem_3.py
```

All simulations use fixed seeds and write their outputs into `assets/`. The code prints its numerical connectivity, convergence, feasibility, and error checks to the console.

## Verification summary

- Problem 1: uses a connected Erdos-Renyi graph with 20 agents, verifies the stable time-step bound, and produces the sequential `DHRUV` GIF.
- Problem 2: verifies graph connectivity and reports consensus disagreement and trajectory RMSE.
- Problem 3: verifies graph connectivity, ADMM primal and dual residuals, integrality, task ownership, and capacities for all three cost cases.
- Problem 4: is derived in the report, including the closed-form ADMM updates for LASSO.
