# PRUEBA CON CLAUDE, NO ES LA VERSIÓN FINAL

import time
from itertools import product
import numpy as np
import cobra
import nashpy as nash

cobra.Configuration().solver = "glpk"   # gurobi/cplex si tienes licencia

# Metabolitos del pool (clave = ID de reacción de intercambio). Ajusta a tus modelos.
POOL = ["EX_glc__D_e", "EX_ac_e", "EX_lac__D_e", "EX_for_e"]

base = cobra.io.load_model("textbook")   # E. coli core, viene con COBRApy

SP = {
    "Bi": dict(model=base.copy(), bio="Biomass_Ecoli_core",
               vmax={"EX_glc__D_e": 8.0}, km={"EX_glc__D_e": 1.0},
               coop_rxns=["EX_ac_e", "EX_for_e"]),
    "Ec": dict(model=base.copy(), bio="Biomass_Ecoli_core",
               vmax={"EX_glc__D_e": 10.0, "EX_ac_e": 5.0, "EX_for_e": 5.0},
               km={"EX_glc__D_e": 1.0, "EX_ac_e": 2.0, "EX_for_e": 2.0},
               coop_rxns=[]),
}
NAMES = list(SP)

STRATS = ["selfish", "coop"]
N_LP = 0  # contador de LPs (métrica de costo)

def uptake_bound(sp, ex, C):
    vmax = sp["vmax"].get(ex, 0.0)
    km = sp["km"].get(ex, 1.0)
    return vmax * C / (km + C)

def fba(sp, C, strat, delta=0.1):
    """Devuelve (mu, flujos de intercambio) para una estrategia."""
    global N_LP
    m = sp["model"]
    with m:                                   # los cambios se revierten al salir
        for ex in POOL:
            if ex in m.reactions:
                m.reactions.get_by_id(ex).lower_bound = -uptake_bound(sp, ex, C[ex])
        bio = m.reactions.get_by_id(sp["bio"])
        m.objective = bio
        mu = m.slim_optimize(); N_LP += 1
        if not np.isfinite(mu) or mu < 1e-9:
            return 0.0, {ex: 0.0 for ex in POOL}
        if strat == "coop" and sp["coop_rxns"]:
            bio.lower_bound = (1 - delta) * mu          # costo de cooperar
            m.objective = {m.reactions.get_by_id(r): 1.0 for r in sp["coop_rxns"]}
            N_LP += 1
        sol = cobra.flux_analysis.pfba(m); N_LP += 1     # evita óptimos degenerados
        fl = {ex: sol.fluxes[ex] if ex in m.reactions else 0.0 for ex in POOL}
        return sol.fluxes[sp["bio"]], fl

def advance(state, strats, dt):
    X, C = state["X"], state["C"]
    res = [fba(SP[n], C, s) for n, s in zip(NAMES, strats)]
    dC = {k: 0.0 for k in C}
    for (mu, fl), x in zip(res, X):
        for k, v in fl.items():
            dC[k] += v * x * dt
    scale = 1.0                                   # no consumir más de lo disponible
    for k in C:
        if dC[k] < 0 and C[k] + dC[k] < 0:
            scale = min(scale, C[k] / (-dC[k] + 1e-12))
    mus = np.array([r[0] for r in res])
    return dict(X=X * np.exp(mus * dt * scale),
                C={k: max(C[k] + dC[k] * scale, 0.0) for k in C})

def payoff_matrices(state, H=5, dt=0.2):
    A = np.zeros((2, 2)); B = np.zeros((2, 2))
    for a, b in product(range(2), range(2)):
        s = state
        for _ in range(H):                        # rollout con estrategias fijas
            s = advance(s, (STRATS[a], STRATS[b]), dt)
        g = np.log(s["X"] / state["X"])
        A[a, b], B[a, b] = g
    return A, B

def nash_strategy(A, B):
    eqs = list(nash.Game(A, B).support_enumeration())
    if not eqs:
        return (0, 0)
    p, q = max(eqs, key=lambda e: e[0] @ A @ e[1] + e[0] @ B @ e[1])
    return int(np.argmax(p)), int(np.argmax(q))   # simplificación: estrategia pura dominante

CACHE = {}

def state_key(state, nbins=4):
    c = np.log10(np.array([state["C"][k] for k in POOL]) + 1e-6)
    r = state["X"][0] / state["X"].sum()
    return tuple(np.round(c * nbins).astype(int)) + (int(round(r * nbins)),)

def get_equilibrium(state, mode, nbins, H, dt):
    if mode == "game_exact":
        return nash_strategy(*payoff_matrices(state, H, dt))
    k = state_key(state, nbins)
    if k not in CACHE:
        CACHE[k] = nash_strategy(*payoff_matrices(state, H, dt))
    return CACHE[k]

def simulate(mode, T=24, dt=0.2, k_resolve=5, nbins=4, H=5):
    """mode: 'selfish' | 'game_exact' | 'game_cache'"""
    global N_LP
    N_LP = 0; t0 = time.time()
    state = dict(X=np.array([0.01, 0.01]),
             C={"EX_glc__D_e": 20.0, "EX_ac_e": 0.0,
                "EX_lac__D_e": 0.0, "EX_for_e": 0.0})
    strats = ("selfish", "selfish")
    traj = []
    for i in range(int(T / dt)):
        if mode != "selfish" and i % k_resolve == 0:
            a, b = get_equilibrium(state, mode, nbins, H, dt)
            strats = (STRATS[a], STRATS[b])
        state = advance(state, strats, dt)
        traj.append((i * dt, *(state["X"] / state["X"].sum())))
    return np.array(traj), dict(time=time.time() - t0, lps=N_LP)

def rmse_vs_data(traj, t_exp, frac_exp):
    sim = np.interp(t_exp, traj[:, 0], traj[:, 1])   # fracción de la especie 1
    return float(np.sqrt(np.mean((sim - frac_exp) ** 2)))

# Ejemplo: Bi parte en 50 % y sube hacia ~75 % en 24 h (curva logística + ruido)
import pandas as pd
print("Empezando...")

rng = np.random.default_rng(42)
t_exp = np.array([0, 2, 4, 6, 8, 12, 16, 20, 24], dtype=float)
frac_exp = np.clip(0.5 - 0.2 / (1 + np.exp(-(t_exp - 10) / 3))
                   + rng.normal(0, 0.02, t_exp.size), 0, 1)

results = {}
for mode in ["selfish", "game_exact", "game_cache"]:
    traj, cost = simulate(mode)
    results[mode] = dict(rmse=rmse_vs_data(traj, t_exp, frac_exp), **cost)
print(pd.DataFrame(results).T)

rows = []
for nbins in [2, 4, 8]:
    for k in [1, 5, 10]:
        CACHE.clear()
        traj, cost = simulate("game_cache", nbins=nbins, k_resolve=k)
        rows.append(dict(nbins=nbins, k=k,
                         rmse=rmse_vs_data(traj, t_exp, frac_exp), **cost))
print(pd.DataFrame(rows))