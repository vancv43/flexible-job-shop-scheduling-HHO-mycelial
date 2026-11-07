# Van Van Chau(1,2), Usa Klinsak(2), Chinda Thorn(2), Anderes Gui(3), Minh Tuan Nguyen(1,*),
# and Phayung Meesad(2,*)
# Nature-Inspired Optimization: Improved Harris Hawks Optimization with Mycelial Search for Flexib
# le Job Shop Scheduling under an Equal Evaluation Budget
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Faculty of Information Technology and Digital Innovation, King Mongkut's University of Technology
#     North Bangkok, Bangkok, Thailand, s6907011966014@email.kmutnb.ac.th,
#     s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (3) School of Information Systems, Bina Nusantara University, Jakarta, Indonesia, anderesgui@binus.ac.id
# (*) Corresponding authors
#
"""Check every value printed in Tables 4-7 and 9 of the paper against the raw runs.

The printed table rows are stored in results/reported_tables.json; every value is recomputed from the run
files at the precision printed in the paper. python verify_results.py, exits 1 on any mismatch
"""
import json
import os
import re
import sys

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
TABLES = json.load(open(os.path.join(CODE, "results", "reported_tables.json")))
R = [json.loads(l) for l in open(os.path.join(CODE, "results", "proposed", "runs.jsonl"))]
A = os.path.join(CODE, "results", "proposed", "analysis")
fr = json.load(open(os.path.join(A, "friedman.json")))
runs = {}
for r in R:
    runs.setdefault((r["instance"], r["algo"]), []).append(r)
MAIN = ["HHO", "BHHO", "WOA", "GWO", "ACO", "IHHO"]
bad = []
n_checked = 0


def check(name, shown, true, tol=0.0):
    global n_checked
    n_checked += 1
    if abs(float(shown) - float(true)) > tol + 1e-9:
        bad.append(f"{name}: shown {shown}, data {true}")


def table_rows(label):
    return TABLES[label]


ub = {}
for label in ["tab:mk", "tab:barnes"]:
    rows = [r for r in table_rows(label) if re.match(r"^[A-Z][A-Z0-9]+$", r[0])]
    assert len(rows) == (10 if label == "tab:mk" else 21)
    for r in rows:
        inst = r[0]
        ub[inst] = float(r[1])
        for k, a in enumerate(MAIN):
            c = [x["cmax"] for x in runs[(inst, a)]]
            assert len(c) == 30
            check(f"{inst} {a} best", r[2 + 2 * k], min(c))
            check(f"{inst} {a} mean", r[3 + 2 * k], round(np.mean(c), 1))
    rp = [r for r in table_rows(label) if r[0].startswith("\\multicolumn{2}{l}{Mean RPD")][0][1:]
    for k in range(12):
        vals = [100 * (float(r[2 + k]) - float(r[1])) / float(r[1]) for r in rows]
        check(f"{label} RPD col {k}", rp[k], round(np.mean(vals), 2))

INV = {"IHHO-MS": "IHHO"}
for r in [r for r in table_rows("tab:stats") if INV.get(r[0], r[0]) in MAIN]:
    a = INV.get(r[0], r[0])
    for col, s in [(5, "main_Brandimarte"), (6, "main_Barnes"), (7, "main")]:
        check(f"rank {a} {s}", r[col], round(fr[s]["avg_rank"][a], 2), 0.005)
cpu = [r for r in table_rows("tab:cpu") if r[0].startswith("Mean CPU")][0][1:]
for k, a in enumerate(MAIN):
    check(f"cpu {a}", cpu[k], round(np.mean([x["cpu_s"] for x in R if x["algo"] == a]), 1))


# Table 9: hawk-free controls and one-at-a-time ablation
import glob  # noqa: E402
NEW = [json.loads(l) for l in open(os.path.join(CODE, "results", "controls", "runs.jsonl"))]
for x in NEW:
    runs.setdefault((x["instance"], x["algo"]), []).append(x)
S11 = json.load(open(os.path.join(CODE, "results", "controls", "analysis", "controls_summary.json")))["summary"]
insts = list(ub)
for r in [r for r in table_rows("tab:controls") if r[0].startswith(("IHHO", "CPLS"))]:
    a = INV.get(r[0], r[0])
    assert all(len(runs[(i, a)]) == 30 for i in insts), a
    for col, pref in [(1, True), (2, False)]:
        ins = [i for i in insts if i.startswith("MK") == pref]
        check(f"controls {a} RPD {pref}", r[col], round(np.mean([100 * (round(np.mean([x["cmax"] for x in runs[(i, a)]]), 1) - ub[i]) / ub[i] for i in ins]), 2))
    if a in S11 and a != "IHHO-plainLS":
        v = S11[a]
        nums = [float(x) for x in re.findall(r"-?\d+\.\d+", r[3])]
        check(f"controls {a} gap", nums[0], round(v["diff"], 2))
        check(f"controls {a} lo", nums[1], round(v["ci95"][0], 2))
        check(f"controls {a} hi", nums[2], round(v["ci95"][1], 2))
        check(f"controls {a} equiv", 1 if r[4].startswith("yes") else 0, int(v["equivalent_05"]))
        w = v["wtl"]
        check(f"controls {a} wtl", 0, 0 if r[5] == f"{w['+']}/{w['=']}/{w['-']}" else 1)
        check(f"controls {a} p", r[6], round(v["signed_rank_p_holm"], 3))

print(f"checked {n_checked} numbers; mismatches: {len(bad)}")
for b in bad:
    print("  ", b)
sys.exit(1 if bad else 0)
