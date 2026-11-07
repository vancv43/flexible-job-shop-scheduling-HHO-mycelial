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
"""Effect sizes, bootstrap intervals, signed-rank details, CPU time and the equal-CPU-time reading, the
distribution of the dive length k_L, the RPD rounding rule and the budget accounting of IHHO-MS.

Reads the existing runs only; no algorithm is re-run.
python extra_stats.py   writes results/proposed/analysis/extra_stats.json and prints a summary
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

CODE = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import analyze as an  # noqa: E402

R = [json.loads(l) for l in open(os.path.join(CODE, "results", "proposed", "runs.jsonl"))]
by = {}
for r in R:
    by.setdefault((r["instance"], r["algo"]), []).append(r)
for v in by.values():
    v.sort(key=lambda r: r["seed"])
INSTS = sorted({r["instance"] for r in R}, key=lambda n: (not n.startswith("MK"), n))
BASE = ["HHO", "BHHO", "WOA", "GWO", "ACO"]
UB = an.UB
CAL = {"MK04", "MK06", "MK10", "MT10XX", "SETB4XY", "SETI5XYZ"}
out = {}


def cm(inst, algo):
    return np.array([r["cmax"] for r in by[(inst, algo)]], float)


def a12(a, b):
    """Vargha-Delaney A: probability that a run of a has a SMALLER makespan than one of b (ties 1/2)."""
    gt = (a[:, None] < b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return (gt + 0.5 * eq) / (a.size * b.size)


def rank_p(a, b):
    """Same rule as analyze.pairwise: scipy ranksums (normal approximation, no tie correction)."""
    if np.all(a == a[0]) and np.array_equal(np.unique(a), np.unique(b)):
        return 1.0
    return float(stats.ranksums(a, b).pvalue)


def rpd(x, inst):
    return 100.0 * (x - UB[inst]) / UB[inst]


# ---------------------------------------------------------------- per-instance tests against the baselines
rows = []
for inst in INSTS:
    a = cm(inst, "IHHO")
    ps = [rank_p(a, cm(inst, o)) for o in BASE]
    adj = an.holm(ps)
    rows.append(dict(instance=inst, **{o: dict(p_holm=float(h), A12=round(float(a12(a, cm(inst, o))), 3))
                                       for o, h in zip(BASE, adj)}))
out["per_instance_baselines"] = rows

# ---------------------------------------------------------------- MS against the plain local search
rng = np.random.default_rng(0)
pl = []
for inst in INSTS:
    a, b = rpd(cm(inst, "IHHO"), inst), rpd(cm(inst, "IHHO-plainLS"), inst)
    boot = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ps = [rank_p(cm(inst, "IHHO"), cm(inst, o)) for o in ["IHHO-noMS", "IHHO-plainLS", "IHHO-randInit"]]
    pl.append(dict(instance=inst, diff=float(a.mean() - b.mean()), lo=float(lo), hi=float(hi),
                   A12=round(float(a12(cm(inst, "IHHO"), cm(inst, "IHHO-plainLS"))), 3),
                   p_raw=ps[1], p_holm=float(an.holm(ps)[1])))
out["per_instance_plainLS"] = pl
d = np.array([x["diff"] for x in pl])
rng2 = np.random.default_rng(1)
bm = [rng2.choice(d, d.size).mean() for _ in range(10000)]
out["plainLS_aggregate"] = dict(
    mean_diff=float(d.mean()), ci=[float(v) for v in np.percentile(bm, [2.5, 97.5])],
    n_ms_better_point=int((d < 0).sum()), n_ci_excl0=int(sum(x["hi"] < 0 or x["lo"] > 0 for x in pl)),
    max_abs_diff=float(np.abs(d).max()), median_A12=float(np.median([x["A12"] for x in pl])))

# ---------------------------------------------------------------- signed-rank tests, exact or approximate
def means(algo, insts):
    return np.array([cm(i, algo).mean() for i in insts])


sr = {}
for name, insts in [("Brandimarte", [i for i in INSTS if i.startswith("MK")]),
                    ("Barnes", [i for i in INSTS if not i.startswith("MK")]), ("All", INSTS)]:
    for o in BASE + ["IHHO-plainLS"]:
        x, y = means("IHHO", insts), means(o, insts)
        dd = x - y
        res = {"n": int(len(insts)), "zeros": int((dd == 0).sum()),
               "ties_abs": int(len(dd[dd != 0]) - len(np.unique(np.abs(dd[dd != 0]))))}
        for meth in ("auto", "exact", "approx"):
            try:
                res[meth] = float(stats.wilcoxon(x, y, method=meth).pvalue)
            except Exception as e:  # noqa: BLE001
                res[meth] = str(e)[:60]
        sr[f"{name}:{o}"] = res
out["signed_rank_methods"] = sr

# ---------------------------------------------------------------- CPU time and the equal-CPU-time reading
cpu = {}
for o in BASE + ["IHHO", "IHHO-plainLS", "IHHO-noMS"]:
    per = [np.mean([r["cpu_s"] for r in by[(i, o)]]) for i in INSTS]
    cv = [np.std([r["cpu_s"] for r in by[(i, o)]], ddof=1) / np.mean([r["cpu_s"] for r in by[(i, o)]])
          for i in INSTS]
    cpu[o] = dict(mean=float(np.mean(per)), min_inst=float(min(per)), max_inst=float(max(per)),
                  median_cv=float(np.median(cv)), max_cv=float(max(cv)))
out["cpu"] = cpu


def truncated(run, frac):
    """Best makespan after floor(frac*50000) decodes, read from the 100-point curve (rounded down)."""
    n = int(math.floor(frac * run["nfe_budget"]))
    k = n * len(run["curve"]) // run["nfe_budget"]
    return run["curve"][k - 1] if k >= 1 else math.inf


eq = {}
for o in BASE:
    wtl = {"+": 0, "=": 0, "-": 0}
    held = {"+": 0, "=": 0, "-": 0}
    fr = []
    for inst in INSTS:
        tm = np.mean([r["cpu_s"] for r in by[(inst, "IHHO")]])
        to = np.mean([r["cpu_s"] for r in by[(inst, o)]])
        T = min(tm, to)
        a = np.array([truncated(r, min(1.0, T / r["cpu_s"])) for r in by[(inst, "IHHO")]], float)
        b = np.array([truncated(r, min(1.0, T / r["cpu_s"])) for r in by[(inst, o)]], float)
        fr.append(T / tm if tm > to else T / to)
        eq.setdefault(inst, {})[o] = (a, b)
    # Holm over the five baselines per instance, as in the main comparison
    out.setdefault("equal_time_fraction", {})[o] = dict(mean=float(np.mean(fr)), min=float(min(fr)),
                                                        slower="IHHO-MS" if np.mean(
                                                            [np.mean([r["cpu_s"] for r in by[(i, "IHHO")]]) >
                                                             np.mean([r["cpu_s"] for r in by[(i, o)]])
                                                             for i in INSTS]) > 0.5 else o)
eqw = {o: {"+": 0, "=": 0, "-": 0} for o in BASE}
eqh = {o: {"+": 0, "=": 0, "-": 0} for o in BASE}
eq_rpd = {o: [] for o in BASE + ["IHHO"]}
for inst in INSTS:
    ps = [rank_p(*eq[inst][o]) for o in BASE]
    adj = an.holm(ps)
    for o, h in zip(BASE, adj):
        a, b = eq[inst][o]
        s = "=" if h >= 0.05 else ("+" if a.mean() < b.mean() else "-")
        eqw[o][s] += 1
        if inst not in CAL:
            eqh[o][s] += 1
        eq_rpd[o].append(rpd(b.mean(), inst))
out["equal_time_wtl"] = eqw
out["equal_time_wtl_held"] = eqh
out["equal_time_mean_rpd_baseline"] = {o: float(np.mean(v)) for o, v in eq_rpd.items() if v}
out["equal_time_mean_rpd_ihho_vs"] = {
    o: float(np.mean([rpd(eq[i][o][0].mean(), i) for i in INSTS])) for o in BASE}

# ---------------------------------------------------------------- distribution of the dive length k_L
rs = np.random.default_rng(0)
beta = 1.5
sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
         / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
u = rs.normal(0, sigma, 10 ** 6)
v = rs.normal(0, 1, 10 ** 6)
L = 0.01 * u / np.abs(v) ** (1 / beta)
kl = {}
for N in (55, 100, 150, 225, 240):
    k = 1 + np.minimum((np.abs(L) * 100).astype(int), N // 10)
    kl[N] = dict(p_k1=float((k == 1).mean()), p_le3=float((k <= 3).mean()), median=float(np.median(k)),
                 q90=float(np.quantile(k, 0.9)), p_cap=float((k == 1 + N // 10).mean()), mean=float(k.mean()))
out["kL"] = kl

# ---------------------------------------------------------------- RPD rounding rule
conv = {}
for o in BASE + ["IHHO", "IHHO-plainLS", "IHHO-noMS", "IHHO-randInit"]:
    for name, pref in (("Brandimarte", True), ("Barnes", False)):
        ins = [i for i in INSTS if i.startswith("MK") == pref]
        rounded = np.mean([rpd(round(cm(i, o).mean(), 1), i) for i in ins])
        exact = np.mean([rpd(cm(i, o).mean(), i) for i in ins])
        conv[f"{o}:{name}"] = [round(float(rounded), 3), round(float(exact), 3)]
out["rpd_convention"] = conv
out["rpd_convention_max_gap"] = float(max(abs(a - b) for a, b in conv.values()))

# ---------------------------------------------------------------- budget accounting
m = [r["myc"] for r in R if r["algo"] == "IHHO"]
fl = np.mean([x["flow"] for x in m])
gr = np.mean([x["growth"] for x in m])
inj = np.mean([x["roi_fire"] for x in m])
out["budget"] = dict(all_runs_exact_50000=all(r["nfe_used"] == 50000 for r in R),
                     flow_per_run=float(fl), growth_per_run=float(gr), roi_fire_per_run=float(inj),
                     flow_equals_tips=all(x["flow"] == x["tips"] for x in m),
                     growth_per_tip=float(np.mean([x["growth"] / x["tips"] for x in m])),
                     iters=float(np.mean([x["iters"] for x in m])))

json.dump(out, open(os.path.join(CODE, "results", "proposed", "analysis", "extra_stats.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("per_instance_baselines", "per_instance_plainLS",
                                                             "rpd_convention")}, indent=1))
