"""
Local capture and leakage in event-driven urban spending.

Models a visitor dollar entering a host-city economy as an absorbing Markov
chain over business and institutional nodes. The quantity of interest is not
whether a dollar "touches" a local business but how many local transactions
it generates before leaving the regional economy.

Two methods are provided and cross-validated:
  1. Analytic solution via the fundamental matrix N = (I - Q)^-1
  2. Monte Carlo simulation of individual dollar trajectories

The transition probabilities are NOT known. They are treated as parameters
and swept, so that conclusions take the form "retention is sensitive to X"
rather than "retention equals Y."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Network specification
# ---------------------------------------------------------------------------

LOCAL_NODES = {
    "local_bar",
    "local_lot",
    "local_wages",
    "local_wholesaler",
    "resident_spending",
    "local_grocery",
}

ABSORBING_NODES = ["exits_region"]


def build_transition_matrix(params: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """
    Build a row-stochastic transition matrix from named parameters.

    Parameters are shares, each in [0, 1]. Complements are computed so every
    row sums to 1 by construction; this is checked before returning.

    Parameter meanings
    ------------------
    p_venue      : share of visitor spending captured at the venue / official channels
    p_bar        : share going to independent food and drink establishments
                   (the independent-parking share is the remainder)
    venue_leak   : share of venue revenue leaving the region immediately
                   (remainder goes to municipal tax)
    bar_wages    : share of independent-business revenue paid as local wages
    bar_supply   : share of independent-business revenue spent on local wholesale
                   (remainder leaks to non-local suppliers)
    wage_local   : share of local wages spent at local establishments
                   (remainder leaves via chain retail, utilities, online)
    wholesale_local : share of wholesaler revenue re-spent locally as wages
                   (remainder leaves to national distributors)
    resident_local  : share of resident income re-spent locally
    tax_local    : share of municipal tax revenue that is spent in-region
    """
    p_venue = params["p_venue"]
    p_bar = params["p_bar"]
    p_lot = 1.0 - p_venue - p_bar
    if p_lot < -1e-9:
        raise ValueError("p_venue + p_bar exceeds 1")
    p_lot = max(p_lot, 0.0)

    venue_leak = params["venue_leak"]
    bar_wages = params["bar_wages"]
    bar_supply = params["bar_supply"]
    wage_local = params["wage_local"]
    wholesale_local = params["wholesale_local"]
    resident_local = params["resident_local"]
    tax_local = params["tax_local"]

    states = [
        "visitor_spend",
        "venue",
        "local_bar",
        "local_lot",
        "local_wages",
        "local_wholesaler",
        "local_grocery",
        "resident_spending",
        "municipal_tax",
        "exits_region",
    ]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))

    def row(src: str, targets: dict[str, float]) -> None:
        for tgt, p in targets.items():
            P[idx[src], idx[tgt]] += p

    # Entry: the visitor dollar splits across venue, independent food/drink,
    # and independent parking.
    row("visitor_spend", {"venue": p_venue, "local_bar": p_bar, "local_lot": p_lot})

    # Venue revenue: mostly out of region (national concessionaires, league
    # revenue sharing, non-local ownership), remainder to municipal tax.
    row("venue", {"exits_region": venue_leak, "municipal_tax": 1.0 - venue_leak})

    # Independent food and drink: wages, local wholesale, and leakage.
    bar_leak = 1.0 - bar_wages - bar_supply
    if bar_leak < -1e-9:
        raise ValueError("bar_wages + bar_supply exceeds 1")
    row(
        "local_bar",
        {
            "local_wages": bar_wages,
            "local_wholesaler": bar_supply,
            "exits_region": max(bar_leak, 0.0),
        },
    )

    # Independent parking: near-total pass-through to resident income.
    row("local_lot", {"resident_spending": 0.9, "municipal_tax": 0.1})

    # Wages: household spending split between local establishments and
    # non-local channels (chain retail, utilities, e-commerce).
    row(
        "local_wages",
        {"local_grocery": wage_local, "exits_region": 1.0 - wage_local},
    )

    # Local grocery / goods: partly local wages again, partly national supply.
    row(
        "local_grocery",
        {"local_wages": 0.35, "exits_region": 0.65},
    )

    # Wholesaler: re-spends some locally as wages, rest to national distributors.
    row(
        "local_wholesaler",
        {"local_wages": wholesale_local, "exits_region": 1.0 - wholesale_local},
    )

    # Resident income: re-spent locally or lost to non-local consumption.
    row(
        "resident_spending",
        {"local_bar": resident_local, "exits_region": 1.0 - resident_local},
    )

    # Municipal revenue: partly spent in-region, partly to state/other.
    row(
        "municipal_tax",
        {"local_wages": tax_local, "exits_region": 1.0 - tax_local},
    )

    # Absorbing state.
    row("exits_region", {"exits_region": 1.0})

    sums = P.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-9):
        bad = [(states[i], s) for i, s in enumerate(sums) if abs(s - 1) > 1e-9]
        raise ValueError(f"rows do not sum to 1: {bad}")

    return P, states


# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------


def fundamental_matrix(P: np.ndarray, states: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Return N = (I - Q)^-1 for the transient states, where Q is the
    transient-to-transient block of P.

    N[i, j] is the expected number of visits to transient state j before
    absorption, starting from transient state i.
    """
    transient = [s for s in states if s not in ABSORBING_NODES]
    ti = [states.index(s) for s in transient]
    Q = P[np.ix_(ti, ti)]
    I = np.eye(len(ti))
    N = np.linalg.inv(I - Q)
    return N, transient


def retention_metrics(params: dict[str, float]) -> dict[str, float]:
    """
    Compute retention metrics exactly, from the fundamental matrix.

    Returns
    -------
    local_transactions : expected number of local-business transactions
        generated by one entering dollar before it leaves the region.
        This is the primary metric: it counts circulation, not mere contact.
    steps_to_exit : expected number of transactions of any kind before exit.
    p_touches_local : probability the dollar reaches at least one local node.
        Reported only for comparison with the naive measure.
    """
    P, states = build_transition_matrix(params)
    N, transient = fundamental_matrix(P, states)
    start = transient.index("visitor_spend")

    visits = {s: N[start, i] for i, s in enumerate(transient)}

    local_transactions = sum(v for s, v in visits.items() if s in LOCAL_NODES)
    steps_to_exit = N[start, :].sum()  # expected transitions before absorption

    # Probability of ever reaching any local node, via absorption into a
    # modified chain where local nodes are made absorbing.
    p_touch = _prob_reach_local(params)

    return {
        "local_transactions": local_transactions,
        "steps_to_exit": steps_to_exit,
        "p_touches_local": p_touch,
    }


def _prob_reach_local(params: dict[str, float]) -> float:
    """Probability of reaching any local node, making local nodes absorbing."""
    P, states = build_transition_matrix(params)
    P = P.copy()
    for s in LOCAL_NODES:
        i = states.index(s)
        P[i, :] = 0.0
        P[i, i] = 1.0

    absorbing = set(LOCAL_NODES) | set(ABSORBING_NODES)
    transient = [s for s in states if s not in absorbing]
    ti = [states.index(s) for s in transient]
    ai = [states.index(s) for s in states if s in absorbing]

    Q = P[np.ix_(ti, ti)]
    R = P[np.ix_(ti, ai)]
    B = np.linalg.inv(np.eye(len(ti)) - Q) @ R

    start = transient.index("visitor_spend")
    absorbing_names = [s for s in states if s in absorbing]
    return sum(
        B[start, j] for j, s in enumerate(absorbing_names) if s in LOCAL_NODES
    )


# ---------------------------------------------------------------------------
# Monte Carlo (used only to validate the analytic result)
# ---------------------------------------------------------------------------


def simulate(params: dict[str, float], n_dollars: int = 200_000, seed: int = 0) -> dict[str, float]:
    """Monte Carlo estimate of the same quantities, for validation."""
    P, states = build_transition_matrix(params)
    rng = np.random.default_rng(seed)
    n = len(states)
    start = states.index("visitor_spend")
    exit_i = states.index("exits_region")
    local_i = {states.index(s) for s in LOCAL_NODES}

    cum = P.cumsum(axis=1)
    local_counts = np.empty(n_dollars)
    steps = np.empty(n_dollars)
    touched = np.empty(n_dollars, dtype=bool)

    for d in range(n_dollars):
        cur = start
        lc = 0
        st = 0
        while cur != exit_i:
            u = rng.random()
            cur = int(np.searchsorted(cum[cur], u))
            st += 1
            if cur in local_i:
                lc += 1
            if st > 10_000:
                break
        local_counts[d] = lc
        steps[d] = st
        touched[d] = lc > 0

    return {
        "local_transactions": float(local_counts.mean()),
        "steps_to_exit": float(steps.mean()),
        "p_touches_local": float(touched.mean()),
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

BASELINE = {
    "p_venue": 0.55,
    "p_bar": 0.30,
    "venue_leak": 0.85,
    "bar_wages": 0.40,
    "bar_supply": 0.30,
    "wage_local": 0.55,
    "wholesale_local": 0.30,
    "resident_local": 0.50,
    "tax_local": 0.40,
}

# Plausible ranges. These are not estimates; they bound the space of values
# a reader might consider defensible.
RANGES = {
    "p_venue": (0.30, 0.70),
    "p_bar": (0.10, 0.45),
    "venue_leak": (0.60, 0.95),
    "bar_wages": (0.25, 0.55),
    "bar_supply": (0.15, 0.45),
    "wage_local": (0.30, 0.75),
    "wholesale_local": (0.10, 0.50),
    "resident_local": (0.25, 0.75),
    "tax_local": (0.10, 0.70),
}


def one_at_a_time(metric: str = "local_transactions", n: int = 25) -> pd.DataFrame:
    """Sweep each parameter across its range holding others at baseline."""
    rows = []
    for name, (lo, hi) in RANGES.items():
        for v in np.linspace(lo, hi, n):
            p = dict(BASELINE)
            p[name] = float(v)
            if p["p_venue"] + p["p_bar"] > 1.0:
                continue
            if p["bar_wages"] + p["bar_supply"] > 1.0:
                continue
            rows.append(
                {"parameter": name, "value": float(v), metric: retention_metrics(p)[metric]}
            )
    return pd.DataFrame(rows)


def elasticity_table(metric: str = "local_transactions") -> pd.DataFrame:
    """
    Range of the metric induced by each parameter over its plausible range,
    holding others at baseline. A compact summary of what matters.
    """
    base = retention_metrics(BASELINE)[metric]
    rows = []
    for name, (lo, hi) in RANGES.items():
        vals = []
        for v in (lo, hi):
            p = dict(BASELINE)
            p[name] = v
            if p["p_venue"] + p["p_bar"] > 1.0 or p["bar_wages"] + p["bar_supply"] > 1.0:
                continue
            vals.append(retention_metrics(p)[metric])
        if len(vals) < 2:
            continue
        rows.append(
            {
                "parameter": name,
                "low": min(vals),
                "high": max(vals),
                "swing": max(vals) - min(vals),
                "swing_pct_of_baseline": 100 * (max(vals) - min(vals)) / base,
            }
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def global_sample(n: int = 20_000, seed: int = 1) -> pd.DataFrame:
    """
    Sample the full parameter space uniformly within ranges and record the
    metric. Used to show the distribution of plausible outcomes rather than
    a single point estimate.
    """
    rng = np.random.default_rng(seed)
    rows = []
    names = list(RANGES)
    while len(rows) < n:
        p = {k: float(rng.uniform(*RANGES[k])) for k in names}
        if p["p_venue"] + p["p_bar"] > 1.0:
            continue
        if p["bar_wages"] + p["bar_supply"] > 1.0:
            continue
        m = retention_metrics(p)
        rows.append({**p, **m})
    return pd.DataFrame(rows)
