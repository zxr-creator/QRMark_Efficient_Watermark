import math
from typing import List, Tuple, Sequence

def so1_allocate_scaling_memory_aware(
    t_baseline: Sequence[float],     # stage times for baseline batch b0 (sec)
    b0: int,                         # baseline batch size used for t_baseline
    B: int,                          # global batch
    mem_per_item: Sequence[int],     # bytes per sample per stage [k]
    M_cap_bytes: int,                # memory cap in bytes
    S_max: int | None = None,        # stream budget (compute streams only)
    s_max_per_stage: Sequence[int] | None = None,
    eps: float = 1e-3,               # accept move threshold
    tau: int = 2,                    # patience
    neigh_mult: Sequence[float] = (0.5, 2/3, 4/3, 2.0),
    alpha: Sequence[float] | None = None,   # scaling efficiency per stage
) :
    """
    Scaling- and memory-aware SO1 (concise).
    T_k(s,b)  ≈  t_k(b0) * (b / b0) / (1 + alpha_k * (s-1))
    Mem cap:  sum_k s_k * m_k * mem_per_item[k]  <= M_cap_bytes
    """
    K = len(t_baseline)
    s = [1] * K
    if s_max_per_stage is None:
        s_max_per_stage = [math.inf] * K
    if alpha is None:
        # heuristics: decode scales best, pre/tile a bit worse
        alpha = [0.75, 0.80, 0.90][:K]

    def T(k: int, sk: int, mk: int) -> float:
        sk = max(1, sk)
        mk = max(1, mk)
        scale = 1.0 + alpha[k] * (sk - 1)
        return t_baseline[k] * (mk / max(1, b0)) / scale

    def J(sv: Sequence[int], mv: Sequence[int]) -> float:
        return max(T(k, sv[k], mv[k]) for k in range(K))

    def feasible(sv: Sequence[int], mv: Sequence[int]) -> bool:
        mem = sum(int(sv[k]) * int(mv[k]) * int(mem_per_item[k]) for k in range(K))
        return mem <= int(M_cap_bytes)

    # --- init: largest uniform m under memory cap ---
    u = sum(s)
    m_unit = max(1, B // u)
    m = [m_unit] * K
    # shrink uniformly until feasible
    while not feasible(s, m) and m_unit > 1:
        m_unit //= 2
        m = [max(1, m_unit) for _ in range(K)]
    # still infeasible? shrink per-stage greedily
    if not feasible(s, m):
        for k in range(K):
            while not feasible(s, m) and m[k] > 1:
                m[k] = max(1, m[k] // 2)

    J_star = J(s, m)
    stall = 0
    used = sum(s)

    # --- coordinate descent: add-stream then tweak m_k ---
    while stall < tau:
        best = None
        gain = 0.0

        # (a) add-one-stream where it helps the minimax most
        if S_max is None or used < S_max:
            for k in range(K):
                if s[k] >= s_max_per_stage[k]:
                    continue
                s2 = list(s); s2[k] += 1
                if feasible(s2, m):
                    d = J_star - J(s2, m)
                    if d > gain + eps:
                        gain, best = d, (s2, list(m))

        # (b) micro-batch equalization in a tiny neighborhood
        for k in range(K):
            for mult in neigh_mult:
                m2 = list(m)
                m2[k] = max(1, int(round(m[k] * mult)))
                if m2[k] == m[k]: 
                    continue
                if feasible(s, m2):
                    d = J_star - J(s, m2)
                    if d > gain + eps:
                        gain, best = d, (list(s), m2)

        if best is None:
            stall += 1
            continue
        # accept best move
        s, m = best
        used = sum(s)
        J_star = J(s, m)
        stall = 0

    # (c) light normalization toward B (don’t break feasibility)
    u = sum(s)
    m_unit = max(1, B // u)
    for k in range(K):
        if T(k, s[k], m[k]) * 1.5 < J_star:  # clearly underutilized
            cand = min(m_unit, m[k] * 2)
            m2 = list(m); m2[k] = max(1, cand)
            if feasible(s, m2):
                m = m2

    return s, m

def allocate_streams_equalized_by_util(
    t_baseline: Sequence[float],     # warm-up stage times, used as default weights
    B: int,                          # global batch upper bound
    mem_per_item: Sequence[int],     # bytes per sample per stage [K]
    M_cap_bytes: int,                # memory upper bound (suggested 0.8 * total_hbm)
    S_max: int,                      # total stream budget (all compute stages)
    util_ratio: Sequence[float] | None = None,  # optional: actual GPU utilization per stage
    b0_for_time: int = 1,            # baseline batch (used for linear time scaling estimate)
) :
    """
    Goal:
      1) Allocate s_k based on GPU utilization ratio (or t_baseline if not provided),
         with sum s_k = S_max and s_k >= 1
      2) Choose a maximum feasible Q <= B, such that for each stage b_k = floor(Q / s_k),
         total memory sum_k s_k * b_k * mem_per_item[k] <= M_cap_bytes
      3) Return (s, b, Q), where b satisfies "b_k * s_k = floor(Q / s_k) * s_k ≈ Q"
    """
    K = len(t_baseline)
    if S_max < K:
        raise ValueError(f"S_max({S_max}) must be >= number of stages({K}) to guarantee s_k >= 1.")

    # ---- 1) Allocate s_k based on utilization ratio (rounding + remainder distribution) ----
    if util_ratio is None:
        # Use stage time proportion as load weight: slower stage → more streams
        util = [max(1e-9, t) for t in t_baseline]
    else:
        util = [max(1e-9, u) for u in util_ratio]

    tot = sum(util)
    raw = [S_max * u / tot for u in util]
    s = [max(1, int(math.floor(x))) for x in raw]
    # Distribute remaining streams to stages with largest fractional parts
    rem = S_max - sum(s)
    if rem > 0:
        frac = sorted([(raw[i] - math.floor(raw[i]), i) for i in range(K)], reverse=True)
        for _, idx in frac[:rem]:
            s[idx] += 1
    # Extreme case: sum(s) > S_max (theoretical; safeguard here)
    while sum(s) > S_max:
        # Subtract 1 from stage with smallest fractional part, but keep >=1
        frac = sorted([(raw[i] - math.floor(raw[i]), i) for i in range(K)])
        for _, idx in frac:
            if s[idx] > 1:
                s[idx] -= 1
                break

    # ---- 2) Choose maximum feasible Q (memory constraint; Q<=B) ----
    def mem_feasible(Q: int) -> bool:
        # b_k = floor(Q / s_k); total mem = sum_k s_k * b_k * mem_k
        b = [max(1, Q // s[k]) for k in range(K)]
        mem = 0
        for k in range(K):
            mem += int(s[k]) * int(b[k]) * int(mem_per_item[k])
            if mem > M_cap_bytes:
                return False
        return True

    lo, hi = K, max(K, B)  # at least each stage can process 1 "parallel bucket"
    best_Q = K
    while lo <= hi:
        mid = (lo + hi) // 2
        if mem_feasible(mid):
            best_Q = mid
            lo = mid + 1
        else:
            hi = mid - 1

    Q = best_Q
    b = [max(1, Q // s[k]) for k in range(K)]  # satisfies b_k * s_k ≈ Q

    return s, b, Q


import math
from typing import List, Sequence, Tuple

def allocate_streams_greedy_exact_flow(
    t_baseline: Sequence[float],     # stage times from warm-up (used as default load weights)
    B: int,                          # global batch upper bound
    mem_per_item: Sequence[int],     # bytes per sample per stage, length=K
    M_cap_bytes: int,                # available memory cap (suggest set to 0.8 * total_hbm)
    S_max: int,                      # total stream budget
    util_ratio: Sequence[float] | None = None,  # optional: actual GPU utilization (if provided, prioritized)
) :
    """
    Return:
      s: number of streams per stage (length K)
      b: micro-batch size per stage (strictly satisfies b_k * s_k = Q)
      Q: strictly equalized flow, Q is a multiple of lcm(s), satisfying Q<=B and memory constraints

    Strategy:
      1) Greedy stream allocation: assign 1 per stage first, then allocate remaining (S_max-K) to stage with max w_k/s_k
         (w_k = util_ratio_k or t_baseline_k)
      2) Let Q = lcm(s) * q, b_k = Q / s_k, memory = Q * sum(mem_per_item)
         Choose largest q such that Q<=B and memory<=M_cap_bytes. If q=0, try reducing streams to lower lcm(s).
    """
    K = len(t_baseline)
    if S_max < K:
        raise ValueError(f"S_max({S_max}) must be >= number of stages({K}).")

    # Load weights: default to stage times, or provided utilization
    w = [float(x) for x in (util_ratio if util_ratio is not None else t_baseline)]
    w = [max(1e-9, x) for x in w]  # avoid 0

    # ----- 1) Greedy allocation of streams -----
    s = [1] * K
    rem = max(0, S_max - K)
    for _ in range(rem):
        k = max(range(K), key=lambda i: w[i] / s[i])
        s[k] += 1

    # Tools: lcm
    def lcm(a: int, b: int) -> int:
        return a // math.gcd(a, b) * b
    def lcm_list(xs: Sequence[int]) -> int:
        v = 1
        for x in xs: v = lcm(v, int(x))
        return v

    # Memory under strict flow: sum_k s_k * b_k * mem_k = Q * sum(mem_k)
    bytes_per_image_all_stages = int(sum(int(m) for m in mem_per_item))

    def choose_q_for_s(s_vec: Sequence[int]):
        """ Given s, return (q, Q, lcm_s). q is max such that Q<=B and Q*sum(mem)<=M_cap_bytes """
        lcm_s = lcm_list(s_vec)
        if lcm_s <= 0:
            return 0, 0, lcm_s
        if bytes_per_image_all_stages <= 0:
            return 0, 0, lcm_s
        q_by_B   = B // lcm_s
        q_by_mem = M_cap_bytes // (lcm_s * bytes_per_image_all_stages)
        q = min(q_by_B, q_by_mem)
        Q = lcm_s * max(0, q)
        return q, Q, lcm_s

    # ----- 2) Choose q/Q to ensure strict flow conservation -----
    q, Q, lcm_s = choose_q_for_s(s)

    # q==0 means B or memory doesn't allow current lcm(s); try reducing streams until feasible
    if q < 1:
        # Try to reduce lcm(s): remove stream from stage with lowest w_i/s_i
        while sum(s) > K:
            cand = [i for i in range(K) if s[i] > 1]
            if not cand:
                break
            i_remove = min(cand, key=lambda i: w[i] / s[i])
            s[i_remove] -= 1
            q, Q, lcm_s = choose_q_for_s(s)
            if q >= 1:
                break
        if q < 1:
            # fallback: 1 stream per stage
            s = [1] * K
            q, Q, lcm_s = choose_q_for_s(s)
            if q < 1:
                # final fallback: set Q=1 (still satisfies strict equality), but very conservative
                Q = 1
                q = 1  # valid when lcm_s=1
    # Compute b per stage, strictly satisfying b_k*s_k = Q
    b = [max(1, Q // sk) for sk in s]
    return s, b, Q

def _lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b

def _lcm_list(xs):
    v = 1
    for x in xs:
        v = _lcm(v, int(x))
    return v

def choose_Q_under_caps(s_vec, B_cap, M_cap_bytes, bytes_per_image_all_stages):
    """
    Given streams s_vec, choose maximum Q = lcm(s)*d, such that:
      1) Q <= B_cap
      2) Q * sum(mem_per_item) <= M_cap_bytes
    Return Q (0 if infeasible) and lcm(s).
    """
    lcm_s = _lcm_list(s_vec)
    if lcm_s <= 0 or bytes_per_image_all_stages <= 0:
        return 0, lcm_s
    X = B_cap // lcm_s                # upper bound for d (only batch constraint)
    if X <= 0:
        return 0, lcm_s
    d_by_mem = M_cap_bytes // (lcm_s * bytes_per_image_all_stages)
    d = min(X, d_by_mem)
    if d <= 0:
        return 0, lcm_s
    return lcm_s * d, lcm_s


def recommend_B_cap(S_max, mem_per_item, M_cap_bytes, B_hint):
    """
    Estimate a more reasonable B_cap (for allocate_streams_greedy_exact_flow):
      1) Use a probe config biased toward Stage-2: s_probe = [1,1,max(1,S_max-2)] to estimate Q_probe
      2) Set B_cap to a nearby multiple of Q_probe (default 4*Q_probe), not exceeding B_hint
      3) If B_hint is not provided (or too small), use 4096 as upper bound
    """
    bytes_per_image_all = int(sum(int(m) for m in mem_per_item))
    s_probe = [1, 1, max(1, int(S_max) - 2)] if int(S_max) >= 3 else [1, 1, 1]
    Q_probe, _ = choose_Q_under_caps(s_probe,
                                     B_hint if B_hint and B_hint > 0 else 4096,
                                     M_cap_bytes,
                                     bytes_per_image_all)
    if Q_probe <= 0:
        s_probe = [1, 1, 1]
        Q_probe, _ = choose_Q_under_caps(s_probe,
                                         B_hint if B_hint and B_hint > 0 else 4096,
                                         M_cap_bytes,
                                         bytes_per_image_all)
    if Q_probe <= 0:
        # fallback: at least return a conservative upper bound
        return min(B_hint if B_hint and B_hint > 0 else 4096, 1024)

    target = 4 * Q_probe  # empirical: 2~4x Q works well, here use 4*Q
    cap = B_hint if B_hint and B_hint > 0 else 4096
    return max(Q_probe, min(cap, target))




def should_fuse_tile_decode(
    t_baseline: Sequence[float],      # [t_pre, t_tile, t_decode]
    s3: Sequence[int],                # streams from a 3-stage greedy plan
    S_max: int,
    fuse_ratio: float = 0.35,         # if t_tile <= fuse_ratio * t_decode → tile is "light"
    min_tile_streams: int = 2,        # if tile gets < this many streams → consider fuse
):
    """
    Decide whether to fuse Stage-1 (tile) and Stage-2 (decode) into a single stage.
    Heuristics:
      - Always fuse when S_max <= 2 (two groups: pre | tile+decode)
      - Or when 3-stage allocation yields s_tile < min_tile_streams and
        t_tile is relatively light vs t_decode (t_tile <= fuse_ratio * t_decode)
    """
    if S_max <= 2:
        return True
    t0, t1, t2 = float(t_baseline[0]), float(t_baseline[1]), float(t_baseline[2])
    s_tile = int(s3[1])
    # tile is light and didn't earn enough streams → fuse
    return (s_tile < int(min_tile_streams)) and (t1 <= fuse_ratio * max(1e-6, t2))


def plan_alloc_with_optional_fuse(
    t_baseline: Sequence[float],     # warm-up times for [pre, tile, decode]
    B: int,
    mem_per_item: Sequence[int],     # bytes per sample for [pre, tile, decode]
    M_cap_bytes: int,
    S_max: int,
    util_ratio: Sequence[float] | None = None,
    fuse_ratio: float = 0.35,
    min_tile_streams: int = 2,
):
    
    # 1) First get a normal 3-stage plan
    s3, b3, Q3 = allocate_streams_greedy_exact_flow(
        t_baseline=t_baseline,
        B=B,
        mem_per_item=mem_per_item,
        M_cap_bytes=M_cap_bytes,
        S_max=S_max,
        util_ratio=util_ratio,
    )

    # ---- NEW: disable fuse when batch size is small ----
    if B < 16:
        return list(s3), list(b3), int(Q3), False

    # 2) Decide fuse or not
    if not should_fuse_tile_decode(t_baseline, s3, S_max, fuse_ratio, min_tile_streams):
        return list(s3), list(b3), int(Q3), False

    # 3) Build a 2-stage problem: stageA=pre, stageB=fused(tile+decode)
    t2 = [float(t_baseline[0]), float(t_baseline[1]) + float(t_baseline[2])]
    m2 = [int(mem_per_item[0]), int(mem_per_item[1]) + int(mem_per_item[2])]

    s2, b2, Q2 = allocate_streams_greedy_exact_flow(
        t_baseline=t2,
        B=B,
        mem_per_item=m2,
        M_cap_bytes=M_cap_bytes,
        S_max=S_max,
        util_ratio=None,  # use t2 as weights
    )
    return list(s2), list(b2), int(Q2), True
