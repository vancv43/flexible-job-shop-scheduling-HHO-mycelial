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
"""FJSP makespan experiments: IHHO-MS (proposed) against HHO, BHHO, GWO, WOA and ACO, plus the ablations.

Arm names: IHHO@nt10_lg80 is IHHO-MS with the calibrated setting (10 tips, 80 decodes each), labelled IHHO
in results/proposed; IHHO is the initial setting (3 tips, 10 decodes), labelled IHHO-init there.
Example: python fjsp_hho.py --data data/brandimarte --instances MK01 --algos IHHO@nt10_lg80 \
         --seeds 0 --nfe 50000 --out out.jsonl

All arms share one instance parser, one active (gap-insertion) decoder, one validity
checker and one budget unit: a fitness evaluation (NFE) = one schedule decode.
Local-search decodes count toward the same NFE budget.
"""
import json
import math
import os
import time
import zlib

import numpy as np


# ----------------------------------------------------------------------------- instance
class Instance:
    def __init__(self, name, n_jobs, n_mach, job_ops, alts):
        self.name = name
        self.n_jobs = n_jobs
        self.n_mach = n_mach
        self.job_ops = job_ops            # job -> list of global op ids, in technological order
        self.alts = alts                  # op -> list of (machine, ptime)
        self.N = len(alts)
        self.op_job = [0] * self.N
        self.op_idx = [0] * self.N
        for j, ops in enumerate(job_ops):
            for k, g in enumerate(ops):
                self.op_job[g] = j
                self.op_idx[g] = k
        # base sequence for random-key decoding: job id repeated once per operation
        self.base_seq = np.array([j for j, ops in enumerate(job_ops) for _ in ops], dtype=np.int64)
        self.n_alt = np.array([len(a) for a in alts], dtype=np.int64)
        self.min_p = [min(p for _, p in a) for a in alts]


def parse_fjs(path):
    toks = open(path).read().split("\n")
    n_jobs, n_mach = map(int, toks[0].split()[:2])
    job_ops, alts = [], []
    for j in range(n_jobs):
        t = list(map(int, toks[1 + j].split()))
        k, i, ops = t[0], 1, []
        for _ in range(k):
            a = t[i]
            i += 1
            pairs = [(t[i + 2 * r], t[i + 2 * r + 1]) for r in range(a)]
            i += 2 * a
            ops.append(len(alts))
            alts.append(pairs)
        assert i == len(t), (path, j)
        job_ops.append(ops)
    # SchedulingLab files are 0-indexed; convert if a 1-indexed file is supplied
    if max(m for a in alts for m, _ in a) == n_mach:
        alts = [[(m - 1, p) for m, p in a] for a in alts]
    name = os.path.splitext(os.path.basename(path))[0].upper()
    return Instance(name, n_jobs, n_mach, job_ops, alts)


# ----------------------------------------------------------------------------- decoder
def decode(inst, os_seq, ms, full=False):
    """Active decoding with gap insertion. os_seq: job ids (operation-based); ms[g]: alt index."""
    nxt = [0] * inst.n_jobs
    ready = [0] * inst.n_jobs
    mach = [[] for _ in range(inst.n_mach)]   # sorted (start, end, op)
    start = [0] * inst.N if full else None
    for j in os_seq:
        j = int(j)
        g = inst.job_ops[j][nxt[j]]
        nxt[j] += 1
        m, p = inst.alts[g][int(ms[g])]
        r = ready[j]
        lst = mach[m]
        prev_end, pos, s = 0, len(lst), None
        for idx, (a, b, _) in enumerate(lst):
            st = r if r > prev_end else prev_end
            if st + p <= a:
                s, pos = st, idx
                break
            prev_end = b
        if s is None:
            s = r if r > prev_end else prev_end
        lst.insert(pos, (s, s + p, g))
        ready[j] = s + p
        if full:
            start[g] = s
    cmax = max(ready)
    if full:
        return cmax, start, mach
    return cmax


def check_schedule(inst, os_seq, ms, cmax):
    """Independent feasibility check of a decoded schedule; raises on violation."""
    c, start, mach = decode(inst, os_seq, ms, full=True)
    assert c == cmax, (c, cmax)
    assert sorted(int(x) for x in os_seq) == sorted(inst.base_seq.tolist())
    end = [0] * inst.N
    for g in range(inst.N):
        assert 0 <= ms[g] < len(inst.alts[g])
        end[g] = start[g] + inst.alts[g][int(ms[g])][1]
    for ops in inst.job_ops:
        for a, b in zip(ops, ops[1:]):
            assert start[b] >= end[a], "precedence"
    for m, lst in enumerate(mach):
        for (a1, b1, g1), (a2, b2, g2) in zip(lst, lst[1:]):
            assert b1 <= a2, "machine overlap"
        for a1, b1, g1 in lst:
            assert inst.alts[g1][int(ms[g1])][0] == m, "machine eligibility"
    assert max(end) == cmax
    return True


def critical_ops(inst, ms, start, mach, cmax):
    """Operations on at least one longest path (start + tail == Cmax)."""
    p = [inst.alts[g][int(ms[g])][1] for g in range(inst.N)]
    msucc = [-1] * inst.N
    order = []
    for lst in mach:
        for (_, _, g1), (_, _, g2) in zip(lst, lst[1:]):
            msucc[g1] = g2
    jsucc = [-1] * inst.N
    for ops in inst.job_ops:
        for a, b in zip(ops, ops[1:]):
            jsucc[a] = b
    order = sorted(range(inst.N), key=lambda g: -start[g])
    tail = [0] * inst.N
    for g in order:
        t = 0
        if jsucc[g] >= 0:
            t = tail[jsucc[g]]
        if msucc[g] >= 0 and tail[msucc[g]] > t:
            t = tail[msucc[g]]
        tail[g] = p[g] + t
    return [g for g in range(inst.N) if start[g] + tail[g] == cmax], msucc


# ----------------------------------------------------------------------------- budget
class Budget:
    def __init__(self, inst, nfe_max, n_points=100):
        self.inst = inst
        self.nfe_max = nfe_max
        self.nfe = 0
        self.best = math.inf
        self.best_sol = None
        self.marks = [int(round(nfe_max * k / n_points)) for k in range(1, n_points + 1)]
        self.curve = []
        self.div_marks = [int(round(nfe_max * k / 20)) for k in range(0, 20)]
        self.div = []                     # (nfe, mean pairwise schedule distance) at 20 points

    def want_div(self):
        return len(self.div) < len(self.div_marks) and self.nfe >= self.div_marks[len(self.div)]

    def record_div(self, sols):
        """Population diversity: mean pairwise sol_distance over the current population."""
        n = len(sols)
        d = [sol_distance(sols[a], sols[b]) for a in range(n) for b in range(a + 1, n)]
        self.div.append((self.nfe, round(float(np.mean(d)), 4) if d else 0.0))

    def left(self):
        return self.nfe < self.nfe_max

    def progress(self):
        return min(1.0, self.nfe / self.nfe_max)

    def eval(self, os_seq, ms):
        c = decode(self.inst, os_seq, ms)
        self.nfe += 1
        if c < self.best:
            self.best = c
            self.best_sol = (np.array(os_seq, copy=True), np.array(ms, copy=True))
        while len(self.curve) < len(self.marks) and self.nfe >= self.marks[len(self.curve)]:
            self.curve.append(self.best)
        return c


# ----------------------------------------------------------------------------- random-key baselines
def rk_decode(inst, x):
    N = inst.N
    os_seq = inst.base_seq[np.argsort(x[:N], kind="stable")]
    ms = np.minimum((x[N:] * inst.n_alt).astype(np.int64), inst.n_alt - 1)
    return os_seq, ms


def rk_eval(bud, x):
    os_seq, ms = rk_decode(bud.inst, x)
    return bud.eval(os_seq, ms)


def levy(rng, d, beta=1.5):
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
             / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, d)
    v = rng.normal(0, 1, d)
    return 0.01 * u / np.abs(v) ** (1 / beta)


def hho_loop(bud, rng, X, lb, ub, evaluate, commit, to_sol):
    """HHO update rules (Heidari et al., 2019). evaluate(x, i) -> (fitness, aux) scores a
    candidate position for hawk i; commit(i, aux) is called when hawk i adopts it."""
    pop, D = X.shape
    F = np.empty(pop)
    for i in range(pop):
        F[i], aux = evaluate(X[i], i)
        commit(i, aux)
    b = int(np.argmin(F))
    rab, frab = X[b].copy(), F[b]
    while bud.left():
        if bud.want_div():
            bud.record_div([to_sol(i) for i in range(pop)])
        E1 = 2 * (1 - bud.progress())
        for i in range(pop):
            if not bud.left():
                break
            E0 = 2 * rng.random() - 1
            E = E1 * E0
            J = 2 * (1 - rng.random())
            if abs(E) >= 1:
                q = rng.random()
                if q >= 0.5:
                    xr = X[rng.integers(pop)]
                    new = xr - rng.random() * np.abs(xr - 2 * rng.random() * X[i])
                else:
                    new = (rab - X.mean(0)) - rng.random() * (lb + rng.random() * (ub - lb))
                X[i] = np.clip(new, lb, ub)
                F[i], aux = evaluate(X[i], i)
                commit(i, aux)
            else:
                r = rng.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    X[i] = np.clip((rab - X[i]) - E * np.abs(J * rab - X[i]), lb, ub)
                    F[i], aux = evaluate(X[i], i)
                    commit(i, aux)
                elif r >= 0.5:
                    X[i] = np.clip(rab - E * np.abs(rab - X[i]), lb, ub)
                    F[i], aux = evaluate(X[i], i)
                    commit(i, aux)
                else:
                    ref = X[i] if abs(E) >= 0.5 else X.mean(0)
                    Y = np.clip(rab - E * np.abs(J * rab - ref), lb, ub)
                    fy, ay = evaluate(Y, i)
                    if fy < F[i]:
                        X[i], F[i] = Y, fy
                        commit(i, ay)
                    elif bud.left():
                        Z = np.clip(Y + rng.random(D) * levy(rng, D), lb, ub)
                        fz, az = evaluate(Z, i)
                        if fz < F[i]:
                            X[i], F[i] = Z, fz
                            commit(i, az)
            if F[i] < frab:
                rab, frab = X[i].copy(), F[i]


def run_hho(inst, bud, rng, pop):
    """Standard HHO on a random-key encoding in [0,1]^(2N)."""
    X = rng.random((pop, 2 * inst.N))
    hho_loop(bud, rng, X, 0.0, 1.0, lambda x, i: (rk_eval(bud, x), None), lambda i, a: None,
             lambda i: rk_decode(inst, X[i]))


def run_bhho(inst, bud, rng, pop, os_bits=8, ms_bits=4):
    """Binary HHO: HHO positions in [-1,1]^Db drive a V-shaped transfer |tanh(x)| that flips
    the hawk's bit string; bits are read as binary random keys (os_bits per OS key, ms_bits
    per MS key), then decoded exactly like the continuous random-key arms."""
    N = inst.N
    Db = N * (os_bits + ms_bits)
    w_os = 2.0 ** np.arange(os_bits - 1, -1, -1)
    w_ms = 2.0 ** np.arange(ms_bits - 1, -1, -1)
    state = rng.integers(0, 2, (pop, Db))

    def keys(bits):
        k_os = (bits[:N * os_bits].reshape(N, os_bits) @ w_os + 0.5) / 2 ** os_bits
        k_ms = (bits[N * os_bits:].reshape(N, ms_bits) @ w_ms + 0.5) / 2 ** ms_bits
        return np.concatenate([k_os, k_ms])

    def evaluate(x, i):
        flip = rng.random(Db) < np.abs(np.tanh(x))
        bits = np.where(flip, 1 - state[i], state[i])
        return rk_eval(bud, keys(bits)), bits

    def commit(i, bits):
        state[i] = bits

    X = rng.uniform(-1, 1, (pop, Db))
    hho_loop(bud, rng, X, -1.0, 1.0, evaluate, commit, lambda i: rk_decode(inst, keys(state[i])))


def run_gwo(inst, bud, rng, pop):
    """Standard GWO (Mirjalili et al., 2014) on the shared random-key encoding."""
    D = 2 * inst.N
    X = rng.random((pop, D))
    F = np.array([rk_eval(bud, x) for x in X], dtype=float)
    while bud.left():
        if bud.want_div():
            bud.record_div([rk_decode(inst, x) for x in X])
        idx = np.argsort(F, kind="stable")
        A_, B_, C_ = X[idx[0]].copy(), X[idx[1]].copy(), X[idx[2]].copy()
        a = 2 * (1 - bud.progress())
        for i in range(pop):
            if not bud.left():
                break
            xs = []
            for lead in (A_, B_, C_):
                A = 2 * a * rng.random(D) - a
                C = 2 * rng.random(D)
                xs.append(lead - A * np.abs(C * lead - X[i]))
            X[i] = np.clip(sum(xs) / 3, 0, 1)
            F[i] = rk_eval(bud, X[i])


def run_woa(inst, bud, rng, pop, b=1.0):
    """Standard WOA (Mirjalili & Lewis, 2016) on the shared random-key encoding."""
    D = 2 * inst.N
    X = rng.random((pop, D))
    F = np.array([rk_eval(bud, x) for x in X], dtype=float)
    k = int(np.argmin(F))
    best, fbest = X[k].copy(), F[k]
    while bud.left():
        if bud.want_div():
            bud.record_div([rk_decode(inst, x) for x in X])
        a = 2 * (1 - bud.progress())
        for i in range(pop):
            if not bud.left():
                break
            A = 2 * a * rng.random() - a
            C = 2 * rng.random()
            if rng.random() < 0.5:
                ref = best if abs(A) < 1 else X[rng.integers(pop)]
                new = ref - A * np.abs(C * ref - X[i])
            else:
                l = rng.uniform(-1, 1)
                new = np.abs(best - X[i]) * math.exp(b * l) * math.cos(2 * math.pi * l) + best
            X[i] = np.clip(new, 0, 1)
            F[i] = rk_eval(bud, X[i])
            if F[i] < fbest:
                best, fbest = X[i].copy(), F[i]


def run_aco(inst, bud, rng, pop, alpha=1.0, beta=2.0, rho=0.1):
    """MAX-MIN ant system building (OS, MS) directly: pheromone on (op, machine) and (step, job)."""
    N, nj = inst.N, inst.n_jobs
    tau_ms = [np.ones(len(a)) for a in inst.alts]
    eta_ms = [np.array([1.0 / p for _, p in a]) ** beta for a in inst.alts]
    tau_os = np.ones((N, nj))
    work = [sum(inst.min_p[g] for g in ops) for ops in inst.job_ops]
    gbest = None
    tmax, tmin = 1.0, 1.0 / (2 * N)
    n_ops = np.array([len(ops) for ops in inst.job_ops])
    op_p = [np.array([inst.min_p[g] for g in ops], dtype=float) for ops in inst.job_ops]
    while bud.left():
        it_best = None
        ants = [] if bud.want_div() else None
        cum_ms = [np.cumsum(tau_ms[g] ** alpha * eta_ms[g]) for g in range(N)]
        tau_os_a = tau_os ** alpha
        for _ in range(pop):
            if not bud.left():
                break
            u = rng.random(N)
            ms = np.array([np.searchsorted(c, ug * c[-1], side="right") for c, ug in zip(cum_ms, u)],
                          dtype=np.int64)
            np.minimum(ms, inst.n_alt - 1, out=ms)
            nxt = np.zeros(nj, dtype=np.int64)
            rem = np.array(work, dtype=float)
            os_seq = np.empty(N, dtype=np.int64)
            v = rng.random(N)
            for s in range(N):
                w = np.where(nxt < n_ops, tau_os_a[s] * (rem + 1.0) ** beta, 0.0)
                c = np.cumsum(w)
                j = min(int(np.searchsorted(c, v[s] * c[-1], side="right")), nj - 1)
                while nxt[j] >= n_ops[j]:      # guard against float edge at the boundary
                    j -= 1
                os_seq[s] = j
                rem[j] -= op_p[j][nxt[j]]
                nxt[j] += 1
            c = bud.eval(os_seq, ms)
            if ants is not None:
                ants.append((os_seq, ms))
            if it_best is None or c < it_best[0]:
                it_best = (c, os_seq, ms)
        if ants:
            bud.record_div(ants)
        if it_best is None:
            break
        if gbest is None or it_best[0] < gbest[0]:
            gbest = it_best
            tmax = 1.0 / (rho * gbest[0])
            tmin = tmax / (2 * N)
        dep = gbest if rng.random() < 0.5 else it_best
        for g in range(N):
            tau_ms[g] *= (1 - rho)
            tau_ms[g][dep[2][g]] += 1.0 / dep[0]
            np.clip(tau_ms[g], tmin, tmax, out=tau_ms[g])
        tau_os *= (1 - rho)
        tau_os[np.arange(N), dep[1]] += 1.0 / dep[0]
        np.clip(tau_os, tmin, tmax, out=tau_os)


# ----------------------------------------------------------------------------- IHHO (proposed)
def init_population(inst, rng, pop, heuristic=True):
    """Multi-modal initialization. MS: global/local/random selection; OS: MWR/MOR/random."""
    N, nj = inst.N, inst.n_jobs
    sols = []
    for k in range(pop):
        if not heuristic:
            ms = rng.integers(0, inst.n_alt)
            os_seq = rng.permutation(inst.base_seq)
            sols.append((os_seq, ms))
            continue
        u = k / pop
        ms = np.empty(N, dtype=np.int64)
        if u < 0.6:      # global selection: least accumulated machine load, random job order
            load = [0] * inst.n_mach
            for j in rng.permutation(nj):
                for g in inst.job_ops[j]:
                    best = min(range(len(inst.alts[g])),
                               key=lambda a: (load[inst.alts[g][a][0]] + inst.alts[g][a][1], rng.random()))
                    ms[g] = best
                    load[inst.alts[g][best][0]] += inst.alts[g][best][1]
        elif u < 0.9:    # local selection: shortest processing time, random tie-break
            for g in range(N):
                ms[g] = min(range(len(inst.alts[g])), key=lambda a: (inst.alts[g][a][1], rng.random()))
        else:
            ms = rng.integers(0, inst.n_alt)
        rule = k % 3
        if rule == 2:
            os_seq = rng.permutation(inst.base_seq)
        else:
            nxt = [0] * nj
            rem_w = [sum(inst.alts[g][ms[g]][1] for g in ops) for ops in inst.job_ops]
            rem_n = [len(ops) for ops in inst.job_ops]
            os_seq = np.empty(N, dtype=np.int64)
            for s in range(N):
                cand = [j for j in range(nj) if rem_n[j] > 0]
                key = rem_w if rule == 0 else rem_n
                top = max(key[j] for j in cand)
                pool = [j for j in cand if key[j] == top]
                j = pool[rng.integers(len(pool))] if rng.random() > 0.2 else cand[rng.integers(len(cand))]
                os_seq[s] = j
                rem_w[j] -= inst.alts[inst.job_ops[j][nxt[j]]][ms[inst.job_ops[j][nxt[j]]]][1]
                rem_n[j] -= 1
                nxt[j] += 1
        sols.append((os_seq, ms))
    return sols


def learn(inst, rng, x, ref, rate):
    """Discrete move of hawk x toward reference ref: each job (OS, POX-style) and each
    operation (MS, uniform) is inherited from ref with probability `rate`."""
    os_x, ms_x = x
    os_r, ms_r = ref
    keep = rng.random(inst.n_jobs) < rate
    fixed = keep[os_r]
    child = np.empty_like(os_x)
    child[fixed] = os_r[fixed]
    child[~fixed] = os_x[~keep[os_x]]
    pick = rng.random(inst.N) < rate
    ms = np.where(pick, ms_r, ms_x)
    return child, ms


def perturb(inst, rng, x, k):
    """k random edits: OS swap of two different jobs or MS reassignment."""
    os_seq, ms = x[0].copy(), x[1].copy()
    for _ in range(k):
        if rng.random() < 0.5:
            i, j = rng.integers(inst.N, size=2)
            os_seq[i], os_seq[j] = os_seq[j], os_seq[i]
        else:
            g = int(rng.integers(inst.N))
            if inst.n_alt[g] > 1:
                ms[g] = (ms[g] + 1 + rng.integers(inst.n_alt[g] - 1)) % inst.n_alt[g]
    return os_seq, ms


def neighbor(inst, rng, sol, move=None):
    """One critical-path move: N1 machine reassignment, N2 swap, N3 insertion of critical ops."""
    os_seq, ms = sol
    cmax, start, mach = decode(inst, os_seq, ms, full=True)
    crit, msucc = critical_ops(inst, ms, start, mach, cmax)
    os_seq, ms = os_seq.copy(), ms.copy()
    if move is None:
        move = rng.integers(3)
    flex = [g for g in crit if inst.n_alt[g] > 1]
    if move == 0 and flex:
        g = flex[rng.integers(len(flex))]
        alts = inst.alts[g]
        others = [a for a in range(len(alts)) if a != ms[g]]
        ms[g] = min(others, key=lambda a: (alts[a][1], rng.random())) if rng.random() < 0.5 \
            else others[rng.integers(len(others))]
        return os_seq, ms
    pos = {}
    cnt = [0] * inst.n_jobs
    for s, j in enumerate(os_seq):
        pos[inst.job_ops[j][cnt[j]]] = s
        cnt[j] += 1
    cset = set(crit)
    mpred = [-1] * inst.N
    for g, h in enumerate(msucc):
        if h >= 0:
            mpred[h] = g
    # (g, h): h directly follows g on a machine, both critical, different jobs
    pairs = [(g, msucc[g]) for g in crit
             if msucc[g] in cset and inst.op_job[g] != inst.op_job[msucc[g]]]
    if not pairs:
        return perturb(inst, rng, (os_seq, ms), 1)
    g, h = pairs[rng.integers(len(pairs))]
    if move == 1:       # N2: put h before its machine predecessor g
        u = g
    else:               # N3: put h before a random earlier op of its critical block
        block = [g]
        while mpred[block[-1]] in cset:
            block.append(mpred[block[-1]])
        block = [u for u in block if inst.op_job[u] != inst.op_job[h]]
        u = block[rng.integers(len(block))]
    if pos[u] >= pos[h]:    # gap insertion already put h after u despite OS order
        return perturb(inst, rng, (os_seq, ms), 1)
    return move_before(os_seq, pos[h], pos[u]), ms


def move_before(os_seq, src, dst):
    """Make the operation at OS position src precede the one at dst (< src). Earlier
    operations of the same job lying in [dst, src) move with it, so the operation keeps
    its identity in the operation-based encoding."""
    if dst >= src:
        return os_seq
    j = os_seq[src]
    seg = os_seq[dst:src + 1]
    mine = seg[seg == j]
    rest = seg[seg != j]
    os_seq[dst:src + 1] = np.concatenate([mine, rest])
    return os_seq


def local_search(inst, bud, rng, sol, f, steps, p_acc):
    cur, fc = sol, f
    best, fb = sol, f
    for _ in range(steps):
        if not bud.left():
            break
        cand = neighbor(inst, rng, cur)
        fn = bud.eval(*cand)
        if fn <= fc or rng.random() < p_acc:
            cur, fc = cand, fn
            if fn < fb:
                best, fb = cand, fn
    return best, fb


def sol_distance(a, b):
    """Normalized distance between two schedules: half OS positions that differ, half MS genes."""
    return 0.5 * float(np.mean(a[0] != b[0])) + 0.5 * float(np.mean(a[1] != b[1]))


def louvain(adj, seed=42, max_levels=10, max_passes=50):
    """Louvain modularity communities (Blondel et al., 2008) of an unweighted graph given as
    adjacency sets. Deterministic for a fixed seed and bounded in levels and passes, so it
    always terminates. Returns a list of node sets."""
    rng = np.random.default_rng(seed)
    W = [{u: 1.0 for u in a} for a in adj]        # weighted adjacency; W[i][i] = 2 x internal weight
    members = [{v} for v in range(len(adj))]
    for _ in range(max_levels):
        n = len(W)
        k = [sum(w.values()) for w in W]
        two_m = sum(k)
        if two_m == 0:
            break
        comm = list(range(n))
        tot = list(k)
        moved_any = False
        for _ in range(max_passes):
            moved = False
            for i in rng.permutation(n):
                i = int(i)
                ci = comm[i]
                tot[ci] -= k[i]
                links = {}
                for u, w in W[i].items():
                    if u != i:
                        links[comm[u]] = links.get(comm[u], 0.0) + w
                best_c = ci
                best_gain = links.get(ci, 0.0) - tot[ci] * k[i] / two_m
                for c in sorted(links):
                    gain = links[c] - tot[c] * k[i] / two_m
                    if gain > best_gain + 1e-12:
                        best_c, best_gain = c, gain
                comm[i] = best_c
                tot[best_c] += k[i]
                if best_c != ci:
                    moved = moved_any = True
            if not moved:
                break
        if not moved_any:
            break
        ids = {c: t for t, c in enumerate(sorted(set(comm)))}
        newW = [dict() for _ in ids]
        newM = [set() for _ in ids]
        for i in range(n):
            a = ids[comm[i]]
            newM[a] |= members[i]
            for u, w in W[i].items():
                b = ids[comm[u]]
                newW[a][b] = newW[a].get(b, 0.0) + w
        W, members = newW, newM
    return members


class Mycelium:
    """Discrete Mycelial Search for FJSP, following the mechanisms of Mycelial Search
    (Dehshibi, 2026, arXiv:2608.23323): hawks are tips and elite schedules are anchors of a
    graph rebuilt every iteration (edge if distance < r_fuse); edge weight e = g/(d+eps) with
    adaptive tip-tip cord conductance g; a filtered Louvain partition sets lambda_intra or
    lambda_inter; a sigmoid gate on the potential difference, with kappa annealed over the
    budget, forms the message weights; the flow moves a tip toward its neighbours through
    `learn`; cords aligned with the flow are reinforced, all tip-tip cords decay; tips with at
    most one neighbour are jittered; ridge-oriented injection spawns from two anchors whose
    relative gap exceeds tau_roi. Hyphal growth along critical-path moves is our addition."""

    def __init__(self, k_anchor=10, r_fuse=0.1, lam_intra=1.0, lam_inter=0.3,
                 kappa_max=10.0, kappa_min=1.0, rho_flow=0.5, eta=0.05, delta=0.02,
                 g_min=0.1, g_max=5.0, tau_align=0.3, tau_roi=0.05, n_min=3, dens_min=0.3, off=()):
        self.k_anchor, self.r_fuse = k_anchor, r_fuse
        self.lam_intra, self.lam_inter = lam_intra, lam_inter
        self.kappa_max, self.kappa_min, self.rho_flow = kappa_max, kappa_min, rho_flow
        self.eta, self.delta, self.g_min, self.g_max = eta, delta, g_min, g_max
        self.tau_align, self.tau_roi = tau_align, tau_roi
        self.n_min, self.dens_min = n_min, dens_min
        self.off = set(off)               # one-at-a-time ablation: "flow", "cord", "inject", "comm"
        self.anchors = []                 # list of ((os, ms), f), best first
        self.g = {}                       # cord conductance of tip pairs (i, j), i < j
        self.st = dict(iters=0, edges=0, comms=0, tips=0, flow=0, flow_improve=0, jitter=0,
                       reinforced=0, growth=0, growth_improve=0, roi_try=0, roi_fire=0, roi_accept=0)

    def summary(self):
        g = np.array(list(self.g.values())) if self.g else np.array([1.0])
        out = dict(self.st)
        out.update(anchors=len(self.anchors), cords=len(self.g),
                   g_q10=round(float(np.quantile(g, 0.1)), 3), g_med=round(float(np.median(g)), 3),
                   g_q90=round(float(np.quantile(g, 0.9)), 3), g_max=round(float(g.max()), 3))
        return out

    def add_anchor(self, sol, f):
        """Insert the incumbent best only if it differs from every retained anchor."""
        for s, _ in self.anchors:
            if np.array_equal(s[0], sol[0]) and np.array_equal(s[1], sol[1]):
                return
        self.anchors.append((sol, f))
        self.anchors.sort(key=lambda t: t[1])
        del self.anchors[self.k_anchor:]

    def communities(self, M, D):
        """Filtered Louvain partition of the fusion graph (own bounded implementation)."""
        adj = [set() for _ in range(M)]
        for p in range(M):
            for q in range(p + 1, M):
                if 0 < D[p, q] < self.r_fuse:   # identical schedules carry no direction
                    adj[p].add(q)
                    adj[q].add(p)
        label = list(range(M))
        n_edges = sum(len(a) for a in adj) // 2
        if "comm" in self.off:            # no communities: every message uses lambda_intra
            return [0] * M, adj
        if n_edges == 0:
            return label, adj
        for c, comm in enumerate(louvain(adj, seed=42)):
            n = len(comm)
            m = sum(1 for v in comm for u in adj[v] if u in comm) // 2
            if n >= self.n_min and 2 * m / (n * (n - 1)) >= self.dens_min:
                for v in comm:
                    label[v] = M + c          # filtered community; others keep singleton labels
        return label, adj

    def step(self, inst, bud, rng, P, F, n_tips, steps, p_acc):
        pop = len(P)
        nodes = [(P[i], F[i]) for i in range(pop)] + self.anchors
        M = len(nodes)
        fv = np.array([f for _, f in nodes], dtype=float)
        span = fv.max() - fv.min()
        pot = np.ones(M) if span == 0 else (fv.max() - fv) / (span + 1e-6)          # Eq. 6
        D = np.zeros((M, M))
        for p in range(M):
            for q in range(p + 1, M):
                D[p, q] = D[q, p] = sol_distance(nodes[p][0], nodes[q][0])
        label, G = self.communities(M, D)
        self.st["iters"] += 1
        self.st["edges"] += sum(len(a) for a in G) // 2
        self.st["comms"] += len({l for l in label if l >= M})
        kappa = self.kappa_max - (self.kappa_max - self.kappa_min) * bud.progress()  # Eq. 10
        for i in np.argsort(F, kind="stable")[:n_tips]:
            i = int(i)
            if not bud.left():
                return
            nb = sorted(G[i])
            # message weights m_ij = lambda * e_ij * sigma_ij (Eqs. 1, 7, 8)
            msg = {}
            for j in nb:
                gij = self.g.setdefault((min(i, j), max(i, j)), 1.0) if j < pop else 1.0
                lam = self.lam_intra if label[i] == label[j] else self.lam_inter
                sig = 1.0 / (1.0 + math.exp(-kappa * (pot[j] - pot[i])))
                msg[j] = lam * gij / (D[i, j] + 1e-6) * sig
            tot = sum(msg.values())
            tip = P[i]
            if tot > 0 and "flow" not in self.off:    # flow: move toward neighbours, heaviest message first
                for j in sorted(msg, key=lambda j: -msg[j]):
                    tip = learn(inst, rng, tip, nodes[j][0], self.rho_flow * msg[j] / tot)
            if len(nb) <= 1 and "flow" not in self.off:   # jitter of weakly connected tips (Eq. 11)
                tip = perturb(inst, rng, tip, 1)
                self.st["jitter"] += 1
            self.st["tips"] += 1
            # cord plasticity on tip-tip edges (Eqs. 2-5). Discrete analogue of the cosine
            # alignment: share of the genes changed by the flow that now carry j's value
            ch_os = tip[0] != P[i][0]
            ch_ms = tip[1] != P[i][1]
            n_ch = ch_os.sum() + ch_ms.sum()
            for j in nb if "cord" not in self.off else ():
                if j >= pop:
                    continue
                key = (min(i, j), max(i, j))
                g = self.g.setdefault(key, 1.0)
                if n_ch > 0:
                    s = (((tip[0] == P[j][0]) & ch_os).sum() + ((tip[1] == P[j][1]) & ch_ms).sum()) / n_ch
                    if s > self.tau_align:
                        g = min(self.g_max, g * (1 + self.eta * s))
                        self.st["reinforced"] += 1
                self.g[key] = max(self.g_min, (1 - self.delta) * g)
            if tip is not P[i]:
                cur, fc = tip, bud.eval(*tip)
                used = 1
                self.st["flow"] += 1
                self.st["flow_improve"] += int(fc < F[i])
            else:
                cur, fc, used = P[i], F[i], 0
            best, fb = (cur, fc) if fc < F[i] else (P[i], F[i])
            # hyphal growth along critical-path moves (our addition)
            for _ in range(steps - used):
                if not bud.left():
                    break
                cand = neighbor(inst, rng, cur)
                fn = bud.eval(*cand)
                self.st["growth"] += 1
                self.st["growth_improve"] += int(fn < fc)
                if fn <= fc or rng.random() < p_acc:
                    cur, fc = cand, fn
                    if fn < fb:
                        best, fb = cand, fn
            P[i], F[i] = best, fb
        # ridge-oriented injection (Eqs. 14-17): midpoint of two anchors with gap > tau_roi,
        # then one critical-path correction from the evaluated midpoint
        if len(self.anchors) >= 2 and bud.left() and "inject" not in self.off:
            a, b = rng.choice(len(self.anchors), 2, replace=False)
            fs = [f for _, f in self.anchors]
            gamma = abs(fs[a] - fs[b]) / (max(fs) - min(fs) + 1e-6)
            self.st["roi_try"] += 1
            if gamma > self.tau_roi:
                self.st["roi_fire"] += 1
                mid = learn(inst, rng, self.anchors[a][0], self.anchors[b][0], 0.5)
                cand, fcand = mid, bud.eval(*mid)
                if bud.left():
                    corr = neighbor(inst, rng, mid)
                    fcorr = bud.eval(*corr)
                    if fcorr < fcand:
                        cand, fcand = corr, fcorr
                w = int(np.argmax(F))
                if fcand < F[w]:
                    P[w], F[w] = cand, fcand
                    self.st["roi_accept"] += 1


def run_ihho(inst, bud, rng, pop, intensify="mycelial", heuristic_init=True,
             ls_top=3, ls_steps=10, p_acc=0.05, myc_kwargs=None, hawks=True):
    """intensify: 'mycelial' (proposed), 'plain' (critical-path local search) or None.
    hawks=False skips the hawk sweep (hawk-free control); the population then changes only
    through intensification and injection."""
    P = init_population(inst, rng, pop, heuristic_init)
    F = [bud.eval(*s) for s in P]
    b = int(np.argmin(F))
    rab, frab = P[b], F[b]
    myc = Mycelium(**(myc_kwargs or {}))
    myc.add_anchor(rab, frab)
    bud.myc = myc if intensify == "mycelial" else None
    while bud.left():
        if bud.want_div():
            bud.record_div(P)
        E1 = 2 * (1 - bud.progress())
        for i in range(pop if hawks else 0):
            if not bud.left():
                break
            E = E1 * (2 * rng.random() - 1)
            aE = abs(E)
            if aE >= 1:                                   # exploration
                if rng.random() >= 0.5:
                    new = learn(inst, rng, P[i], P[rng.integers(pop)], 0.5)
                else:
                    new = perturb(inst, rng, learn(inst, rng, P[i], rab, 0.5), 2)
                P[i], F[i] = new, bud.eval(*new)
            else:
                r = rng.random()
                if r >= 0.5:                              # soft (|E|>=0.5) / hard besiege
                    new = learn(inst, rng, P[i], rab, 1 - aE)
                    if aE < 0.5:
                        new = perturb(inst, rng, new, 1)
                    P[i], F[i] = new, bud.eval(*new)
                else:                                     # progressive rapid dives, greedy
                    Y = learn(inst, rng, P[i], rab, 1 - aE)
                    fy = bud.eval(*Y)
                    if fy < F[i]:
                        P[i], F[i] = Y, fy
                    elif bud.left():
                        k = 1 + min(int(abs(levy(rng, 1)[0]) * 100), inst.N // 10)
                        Z = perturb(inst, rng, Y, k)
                        fz = bud.eval(*Z)
                        if fz < F[i]:
                            P[i], F[i] = Z, fz
            if F[i] < frab:
                rab, frab = P[i], F[i]
                myc.add_anchor(rab, frab)
        if intensify == "mycelial":
            myc.step(inst, bud, rng, P, F, ls_top, ls_steps, p_acc)
        elif intensify == "plain":
            for i in np.argsort(F, kind="stable")[:ls_top]:
                if not bud.left():
                    break
                P[i], F[i] = local_search(inst, bud, rng, P[i], F[i], ls_steps, p_acc)
        b = int(np.argmin(F))
        if F[b] < frab:
            rab, frab = P[b], F[b]
            myc.add_anchor(rab, frab)


ALGOS = {
    "IHHO": lambda i, b, r, p: run_ihho(i, b, r, p),
    "HHO": run_hho,
    "BHHO": run_bhho,
    "GWO": run_gwo,
    "WOA": run_woa,
    "ACO": run_aco,
    "IHHO-noMS": lambda i, b, r, p: run_ihho(i, b, r, p, intensify=None),
    "IHHO-plainLS": lambda i, b, r, p: run_ihho(i, b, r, p, intensify="plain"),
    "IHHO-randInit": lambda i, b, r, p: run_ihho(i, b, r, p, heuristic_init=False),
    # controls of the ablation study (appended so earlier RNG stream codes are unchanged); all at the proposed (10, 80)
    "IHHO-MS-noHawk": lambda i, b, r, p: run_ihho(i, b, r, p, ls_top=10, ls_steps=80, hawks=False),
    "CPLS-elite": lambda i, b, r, p: run_ihho(i, b, r, p, intensify="plain", ls_top=10, ls_steps=80,
                                              hawks=False),
    "IHHO-MS-noFlow": lambda i, b, r, p: run_ihho(i, b, r, p, ls_top=10, ls_steps=80,
                                                  myc_kwargs={"off": ("flow",)}),
    "IHHO-MS-noCord": lambda i, b, r, p: run_ihho(i, b, r, p, ls_top=10, ls_steps=80,
                                                  myc_kwargs={"off": ("cord",)}),
    "IHHO-MS-noInject": lambda i, b, r, p: run_ihho(i, b, r, p, ls_top=10, ls_steps=80,
                                                    myc_kwargs={"off": ("inject",)}),
    "IHHO-MS-noComm": lambda i, b, r, p: run_ihho(i, b, r, p, ls_top=10, ls_steps=80,
                                                  myc_kwargs={"off": ("comm",)}),
}
ALGO_CODE = {k: n for n, k in enumerate(ALGOS)}


def sensitivity_arm(algo):
    """'IHHO@nt3_lg10' or 'IHHO@rf0.1_pf0.5' -> IHHO with those settings, the calibration grids;
    'IHHO-plainLS@nt10_lg80' and 'IHHO-randInit@nt10_lg80' apply them to an ablation arm."""
    base = algo.split("@", 1)[0]
    kw = {"IHHO": {}, "IHHO-plainLS": {"intensify": "plain"}, "IHHO-randInit": {"heuristic_init": False}}[base]
    kw, myc = dict(kw), {}
    for part in algo.split("@", 1)[1].split("_"):
        key, val = part[:2], part[2:]
        if key == "nt":
            kw["ls_top"] = int(val)
        elif key == "lg":
            kw["ls_steps"] = int(val)
        elif key == "rf":
            myc["r_fuse"] = float(val)
        elif key == "pf":
            myc["rho_flow"] = float(val)
        else:
            raise ValueError(algo)
    return lambda i, b, r, p: run_ihho(i, b, r, p, myc_kwargs=myc, **kw)


def run_one(path, algo, seed, nfe, pop):
    inst = parse_fjs(path)
    inst_no = int(inst.name[2:]) if inst.name.startswith("MK") else zlib.crc32(inst.name.encode())
    fn = sensitivity_arm(algo) if "@" in algo else ALGOS[algo]
    rng = np.random.default_rng([seed, inst_no, ALGO_CODE[algo.split("@")[0]]])
    bud = Budget(inst, nfe)
    t0 = time.process_time()
    fn(inst, bud, rng, pop)
    cpu = time.process_time() - t0
    os_seq, ms = bud.best_sol
    check_schedule(inst, os_seq, ms, bud.best)
    while len(bud.curve) < len(bud.marks):
        bud.curve.append(bud.best)
    return {"instance": inst.name, "algo": algo, "seed": seed, "nfe_budget": nfe, "pop": pop,
            "nfe_used": bud.nfe, "cmax": int(bud.best), "cpu_s": round(cpu, 3),
            "curve": [int(c) for c in bud.curve], "div": bud.div,
            "myc": bud.myc.summary() if getattr(bud, "myc", None) else None,
            "best_os": os_seq.tolist(), "best_ms": ms.tolist()}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--instances", default="MK01")
    ap.add_argument("--algos", default=",".join(ALGOS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--nfe", type=int, default=20000)
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    with open(a.out, "a") as fh:
        for name in a.instances.split(","):
            for algo in a.algos.split(","):
                for s in seeds:
                    res = run_one(os.path.join(a.data, name.lower() + ".fjs"), algo, s, a.nfe, a.pop)
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    print(res["instance"], algo, s, res["cmax"], res["cpu_s"], flush=True)
