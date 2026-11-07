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
"""Hawk-free controls and one-at-a-time ablation of MS (Table 9 of the paper), analysis plan fixed before
the runs: Holm over the six variants, signed-rank on instance means, bootstrap over instances and a
two one-sided test with a +-0.5 RPD-point equivalence margin.
Usage: python analyze_controls.py -> results/controls/analysis/controls_summary.json
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

import analyze as an

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["IHHO-MS-noHawk", "CPLS-elite", "IHHO-MS-noFlow", "IHHO-MS-noCord", "IHHO-MS-noInject", "IHHO-MS-noComm"]
OUT = os.path.join(HERE, "results", "controls", "analysis")
os.makedirs(OUT, exist_ok=True)

ref = {}
for l in open(os.path.join(HERE, "results", "proposed", "runs.jsonl")):
    r = json.loads(l)
    if r["algo"] in ("IHHO", "IHHO-plainLS"):
        ref.setdefault((r["instance"], r["algo"]), {})[r["seed"]] = r
new = {}
for f in [os.path.join(HERE, "results", "controls", "runs.jsonl")]:
    for l in open(f):
        r = json.loads(l)
        k = (r["instance"], r["algo"])
        assert r["seed"] not in new.setdefault(k, {}), ("duplicate", k, r["seed"])
        new[k][r["seed"]] = r
INSTS = sorted({i for i, _ in ref}, key=lambda n: (not n.startswith("MK"), n))
# validation: every run present once, budget exact
missing = [(i, a, s) for i in INSTS for a in ARMS for s in range(30) if s not in new.get((i, a), {})]
bad = [(k, s) for k, v in new.items() for s, r in v.items() if r["nfe_used"] != 50000 or r["algo"] not in ARMS]
print("runs", sum(len(v) for v in new.values()), "missing", len(missing), "bad", len(bad))
if missing or bad:
    print(missing[:5], bad[:5])
    if "--partial" not in sys.argv:
        sys.exit(1)


def cm(d, i, a):
    return np.array([d[(i, a)][s]["cmax"] for s in range(30) if s in d.get((i, a), {})], float)


def rpd(x, i):
    return 100 * (x - an.UB[i]) / an.UB[i]


def a12(a, b):
    return ((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (a.size * b.size)


def p_rs(a, b):
    if np.all(a == a[0]) and np.array_equal(np.unique(a), np.unique(b)):
        return 1.0
    return float(stats.ranksums(a, b).pvalue)


arms = [a for a in ARMS if all((i, a) in new for i in INSTS)]
per, wtl = [], {a: {"+": 0, "=": 0, "-": 0} for a in arms}
for i in INSTS:
    x = cm(ref, i, "IHHO")
    ps = [p_rs(x, cm(new, i, a)) for a in arms]
    for a, p, h in zip(arms, ps, an.holm(ps)):
        y = cm(new, i, a)
        s = "=" if h >= 0.05 else ("+" if x.mean() < y.mean() else "-")
        wtl[a][s] += 1
        per.append(dict(instance=i, arm=a, mean_ms=x.mean(), mean_arm=y.mean(), p_holm=float(h),
                        A12=float(a12(x, y)), sign=s))
summary = {}
rng = np.random.default_rng(0)
comp = arms + ["IHHO-plainLS"]
srp = []
for a in comp:
    src = ref if a == "IHHO-plainLS" else new
    d = np.array([rpd(cm(ref, i, "IHHO").mean(), i) - rpd(cm(src, i, a).mean(), i) for i in INSTS])
    bm = np.array([rng.choice(d, d.size).mean() for _ in range(10000)])
    lo95, hi95 = np.percentile(bm, [2.5, 97.5])
    lo90, hi90 = np.percentile(bm, [5, 95])
    mx = np.array([cm(ref, i, "IHHO").mean() for i in INSTS])
    my = np.array([cm(src, i, a).mean() for i in INSTS])
    srp.append(1.0 if np.all(mx == my) else float(stats.wilcoxon(mx, my).pvalue))
    summary[a] = dict(mean_rpd_arm=float(np.mean([rpd(cm(src, i, a).mean(), i) for i in INSTS])),
                      diff=float(d.mean()), ci95=[float(lo95), float(hi95)], ci90=[float(lo90), float(hi90)],
                      equivalent_05=bool(-0.5 < lo90 and hi90 < 0.5), wtl=wtl.get(a),
                      median_A12=float(np.median([p["A12"] for p in per if p["arm"] == a])) if a in arms else None)
for a, p, h in zip(comp, srp, an.holm(srp[:len(arms)]).tolist() + [srp[-1]]):
    summary[a]["signed_rank_p"] = p
    summary[a]["signed_rank_p_holm"] = h
summary["IHHO-MS mean RPD"] = float(np.mean([rpd(cm(ref, i, "IHHO").mean(), i) for i in INSTS]))
json.dump(dict(summary=summary, per_instance=per), open(os.path.join(OUT, "controls_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
