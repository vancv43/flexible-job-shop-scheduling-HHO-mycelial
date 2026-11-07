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
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fjsp_hho as F

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "brandimarte")


def toy():
    # 2 jobs x 2 machines; job0: O00 {m0:3, m1:5}, O01 {m1:2}; job1: O10 {m0:2}, O11 {m0:4, m1:1}
    txt = "2 2\n2 2 0 3 1 5 1 1 2\n2 1 0 2 2 0 4 1 1\n"
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mk99.fjs")
    open(p, "w").write(txt)
    return F.parse_fjs(p)


def test_decoder_hand_computed():
    inst = toy()
    # OS = [0,1,0,1], MS = [m0, m1 | m0, m1]
    # O00 m0 [0,3]; O10 m0 [3,5]; O01 m1 [3,5]; O11 m1 [5,6] -> Cmax 6
    c = F.decode(inst, np.array([0, 1, 0, 1]), np.array([0, 0, 0, 1]))
    assert c == 6
    assert F.check_schedule(inst, np.array([0, 1, 0, 1]), np.array([0, 0, 0, 1]), 6)


def test_gap_insertion():
    inst = toy()
    # OS = [1,0,...]: O10 m0 [0,2]; O00 m1 (5) [0,5]; O11 on m0 (4) [2,6]; O01 m1 [5,7] -> 7
    c = F.decode(inst, np.array([1, 0, 1, 0]), np.array([1, 0, 0, 0]))
    assert c == 7


def test_instance_sizes():
    sizes = {1: 55, 2: 58, 3: 150, 4: 90, 5: 106, 6: 150, 7: 100, 8: 225, 9: 240, 10: 240}
    for k, n in sizes.items():
        assert F.parse_fjs(os.path.join(DATA, "mk%02d.fjs" % k)).N == n


def test_all_arms_feasible_and_deterministic():
    p = os.path.join(DATA, "mk01.fjs")
    for algo in F.ALGOS:
        a = F.run_one(p, algo, 7, 600, 10)
        b = F.run_one(p, algo, 7, 600, 10)
        assert a["cmax"] == b["cmax"] and a["curve"] == b["curve"], algo
        assert a["nfe_used"] <= 600 + 1 and len(a["curve"]) == 100
        assert a["cmax"] >= 36  # trivial lower bound check (BKS 40; LB 36)


def op_positions(inst, os_seq):
    pos, cnt = {}, [0] * inst.n_jobs
    for s, j in enumerate(os_seq):
        pos[inst.job_ops[j][cnt[j]]] = s
        cnt[j] += 1
    return pos


def test_move_before_keeps_operation_identity():
    inst = F.parse_fjs(os.path.join(DATA, "mk04.fjs"))
    rng = np.random.default_rng(0)
    for _ in range(300):
        os_seq = rng.permutation(inst.base_seq)
        src, dst = sorted(rng.choice(inst.N, 2, replace=False))[::-1]
        pos = op_positions(inst, os_seq)
        by_pos = {s: g for g, s in pos.items()}
        h, u = by_pos[src], by_pos[dst]
        if inst.op_job[h] == inst.op_job[u]:
            continue
        new = F.move_before(os_seq.copy(), src, dst)
        p2 = op_positions(inst, new)
        assert p2[h] < p2[u]                                   # h now precedes u
        assert sorted(new.tolist()) == sorted(os_seq.tolist())
        others = [g for g in range(inst.N) if inst.op_job[g] not in (inst.op_job[h],)]
        order_before = sorted(others, key=lambda g: pos[g])
        order_after = sorted(others, key=lambda g: p2[g])
        assert order_before == order_after                     # other jobs keep relative order


def test_louvain_matches_networkx_quality_and_terminates():
    import networkx as nx
    from networkx.algorithms.community import modularity, louvain_communities
    rng = np.random.default_rng(3)
    for t in range(40):
        n = int(rng.integers(5, 45))
        G = nx.gnp_random_graph(n, float(rng.uniform(0.05, 0.5)), seed=int(rng.integers(1e6)))
        if G.number_of_edges() == 0:
            continue
        adj = [set(G[v]) for v in range(n)]
        ours = F.louvain(adj, seed=42)
        assert sorted(v for c in ours for v in c) == list(range(n))      # a partition
        q_ours = modularity(G, ours)
        q_nx = modularity(G, louvain_communities(G, seed=42))
        assert q_ours >= q_nx - 0.05, (t, q_ours, q_nx)
    # two cliques joined by one edge -> two communities
    G = nx.barbell_graph(6, 0)
    ours = F.louvain([set(G[v]) for v in range(12)], seed=42)
    assert sorted(map(sorted, ours)) == [list(range(6)), list(range(6, 12))]
