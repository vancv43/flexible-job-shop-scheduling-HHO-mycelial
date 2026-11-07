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
"""Statistics of the two calibration grids (Figure 12 of the paper).

Reads results/calibration/runs.jsonl (six instances, seeds 100-109), ranks the 25 cells of each grid per
instance by mean final makespan, averages the ranks over the six instances, and runs the Friedman test across
the cells together with Kendall's W. Writes results/calibration/summary.json.
Usage: python calibration_stats.py
"""
import collections
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "calibration", "runs.jsonl")
OUT = os.path.join(HERE, "results", "calibration", "summary.json")
INSTS = ["MK04", "MK06", "MK10", "MT10XX", "SETB4XY", "SETI5XYZ"]
GRIDS = {
    "A": dict(rows=("nt", [1, 2, 3, 5, 10]), cols=("lg", [5, 10, 20, 40, 80]), initial=(3, 10),
              title="number of tips n_tip x growth length L_g"),
    "B": dict(rows=("rf", [0.05, 0.1, 0.2, 0.3, 0.5]), cols=("pf", [0.1, 0.25, 0.5, 0.75, 1.0]),
              initial=(0.1, 0.5), title="fusion radius r_fuse x flow rate rho_f"),
}


def main():
    R = collections.defaultdict(list)
    for line in open(SRC):
        r = json.loads(line)
        R[(r["instance"], r["algo"])].append(r["cmax"])
    out = {}
    for key, g in GRIDS.items():
        rv, cv = g["rows"][1], g["cols"][1]
        cells = [(a, b) for a in rv for b in cv]
        arm = lambda a, b: f"IHHO@{g['rows'][0]}{a}_{g['cols'][0]}{b}"  # noqa: E731
        missing = [(i, arm(a, b)) for i in INSTS for a, b in cells if len(R[(i, arm(a, b))]) != 10]
        assert not missing, f"{len(missing)} incomplete cells, e.g. {missing[:3]}"
        means = np.array([[np.mean(R[(i, arm(a, b))]) for a, b in cells] for i in INSTS])
        ranks = np.array([stats.rankdata(row) for row in means])
        mr = ranks.mean(0).reshape(len(rv), len(cv))
        fr = stats.friedmanchisquare(*[means[:, k] for k in range(len(cells))])
        best = np.unravel_index(np.argmin(mr), mr.shape)
        ini = (rv.index(g["initial"][0]), cv.index(g["initial"][1]))
        out[key] = dict(grid=g["title"], rows=rv, cols=cv, best=[rv[best[0]], cv[best[1]]],
                        best_rank=round(float(mr[best]), 2), initial=list(g["initial"]),
                        initial_rank=round(float(mr[ini]), 2),
                        initial_position=int(stats.rankdata(mr.ravel())[ini[0] * len(cv) + ini[1]]),
                        friedman_p=float(fr.pvalue),
                        kendall_w=round(float(fr.statistic / (len(INSTS) * (len(cells) - 1))), 3),
                        mean_rank=mr.round(2).tolist())
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "mean_rank"} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
