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
"""Per-instance statistics, Holm-corrected rank-sum tests and Friedman ranks of the main comparison.

Usage: python analyze.py results/proposed results/proposed/analysis
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

UB = {"MK01": 40, "MK02": 26, "MK03": 204, "MK04": 60, "MK05": 172,
      "MK06": 58, "MK07": 139, "MK08": 523, "MK09": 307, "MK10": 197}
_B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "barnes", "BOUNDS.txt")
if os.path.exists(_B):
    for _line in open(_B):
        _n, _j, _m, _opt = _line.split()
        UB[_n.upper()] = int(_opt)
SET = lambda name: "Brandimarte" if name.startswith("MK") else "Barnes"
MAIN = ["HHO", "BHHO", "GWO", "WOA", "ACO", "IHHO"]
ABL = ["IHHO-noMS", "IHHO-plainLS", "IHHO-randInit"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load(src):
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "**", "*.jsonl"), recursive=True)):
        for line in open(f):
            r = json.loads(line)
            rows.append({k: r[k] for k in ("instance", "algo", "seed", "cmax", "cpu_s", "nfe_used")})
    df = pd.DataFrame(rows).drop_duplicates(["instance", "algo", "seed"])
    return df


def per_instance(df):
    g = df.groupby(["instance", "algo"])["cmax"]
    t = g.agg(Best="min", Mean="mean", Std="std", Worst="max", n="count").reset_index()
    t["CPU"] = df.groupby(["instance", "algo"])["cpu_s"].mean().values
    t["UB"] = t["instance"].map(UB)
    t["RPD_best"] = 100 * (t["Best"] - t["UB"]) / t["UB"]
    t["RPD_mean"] = 100 * (t["Mean"] - t["UB"]) / t["UB"]
    return t


def pairwise(df, ref, others):
    out = []
    for inst in sorted(df["instance"].unique()):
        a = df[(df.instance == inst) & (df.algo == ref)]["cmax"].values
        ps, rows = [], []
        for o in others:
            b = df[(df.instance == inst) & (df.algo == o)]["cmax"].values
            if len(a) == 0 or len(b) == 0:
                continue
            p = 1.0 if np.all(a == a[0]) and np.array_equal(np.unique(a), np.unique(b)) \
                else stats.ranksums(a, b).pvalue
            ps.append(p)
            rows.append([inst, ref, o, np.median(a), np.median(b), p])
        if not rows:
            continue
        for r, pa in zip(rows, holm(ps)):
            sign = "=" if pa >= 0.05 else ("+" if r[3] < r[4] or (r[3] == r[4] and
                   df[(df.instance == inst) & (df.algo == ref)]["cmax"].mean() <
                   df[(df.instance == inst) & (df.algo == r[2])]["cmax"].mean()) else "-")
            out.append(r + [pa, sign])
    return pd.DataFrame(out, columns=["instance", "ref", "other", "med_ref", "med_other",
                                      "p_raw", "p_holm", "sign"])


def friedman(t, arms):
    m = t[t.algo.isin(arms)].pivot(index="instance", columns="algo", values="Mean")[arms].dropna()
    ranks = m.rank(axis=1).mean().sort_values()
    res = {"n_instances": int(len(m)), "avg_rank": ranks.round(3).to_dict()}
    if len(m) >= 2:
        res["friedman_p"] = float(stats.friedmanchisquare(*[m[a] for a in arms]).pvalue)
        ps, names = [], []
        for a in arms:
            if a == "IHHO":
                continue
            d = m["IHHO"] - m[a]
            ps.append(1.0 if np.all(d == 0) else float(stats.wilcoxon(m["IHHO"], m[a]).pvalue))
            names.append(a)
        res["signed_rank_vs_IHHO"] = {n: {"p_raw": p, "p_holm": float(h)}
                                      for n, p, h in zip(names, ps, holm(ps))}
    return res


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    df = load(src)
    df.to_csv(os.path.join(dst, "runs.csv"), index=False)
    t = per_instance(df)
    t.to_csv(os.path.join(dst, "per_instance.csv"), index=False)
    pw = pairwise(df, "IHHO", [a for a in MAIN if a != "IHHO"])
    pw.to_csv(os.path.join(dst, "wilcoxon_main.csv"), index=False)
    pa = pairwise(df, "IHHO", ABL)
    pa.to_csv(os.path.join(dst, "wilcoxon_ablation.csv"), index=False)
    fr = {"main": friedman(t, MAIN), "with_ablation": friedman(t, MAIN + ABL)}
    for setname in ("Brandimarte", "Barnes"):
        ts = t[t.instance.map(SET) == setname]
        if len(ts):
            fr["main_" + setname] = friedman(ts, MAIN)
    json.dump(fr, open(os.path.join(dst, "friedman.json"), "w"), indent=2)
    wtl = pw.groupby(["other", "sign"]).size().unstack(fill_value=0)
    wtl.to_csv(os.path.join(dst, "win_tie_loss.csv"))
    print(t.pivot(index="instance", columns="algo", values="Mean").round(2).to_string())
    print(wtl.to_string())
    print(json.dumps(fr, indent=2))
