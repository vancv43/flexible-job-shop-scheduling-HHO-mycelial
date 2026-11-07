#!/usr/bin/env bash
# Rebuild every analysis file from the raw runs and check the reported table values (about one minute).
# The commented block at the end reruns the experiments themselves; that takes many CPU-hours.
set -euo pipefail
PY=${PYTHON:-python}
cd "$(dirname "$0")"
$PY analyze.py results/proposed results/proposed/analysis   # Tables 4-6, 8, 11 and per-instance tests
$PY extra_stats.py                                          # Table 7, Table A2, effect sizes, equal CPU time
$PY analyze_controls.py                                     # Table 9
$PY calibration_stats.py                                    # Figure 12, calibration grids
$PY verify_results.py                                       # every value of Tables 4-7 and 9
$PY -m pytest -q tests                                      # decoder, schedule checker and moves

# Full experiments, one line per arm; each line writes 31 instances x 30 seeds.
# B="MK01,MK02,MK03,MK04,MK05,MK06,MK07,MK08,MK09,MK10"
# R="MT10C1,MT10CC,MT10X,MT10XX,MT10XXX,MT10XY,MT10XYZ,SETB4C9,SETB4CC,SETB4X,SETB4XX,SETB4XXX,SETB4XY,SETB4XYZ,SETI5C12,SETI5CC,SETI5X,SETI5XX,SETI5XXX,SETI5XY,SETI5XYZ"
# S=$(seq -s, 0 29)
# for A in HHO BHHO GWO WOA ACO IHHO IHHO-noMS IHHO@nt10_lg80 IHHO-plainLS@nt10_lg80 IHHO-randInit@nt10_lg80 \
#          IHHO-MS-noHawk CPLS-elite IHHO-MS-noFlow IHHO-MS-noCord IHHO-MS-noInject IHHO-MS-noComm; do
#   $PY fjsp_hho.py --data data/brandimarte --instances $B --algos $A --seeds $S --nfe 50000 --out out_$A.jsonl
#   $PY fjsp_hho.py --data data/barnes --instances $R --algos $A --seeds $S --nfe 50000 --out out_$A.jsonl
# done
