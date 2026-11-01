"""
SteadyCom con inercia metabólica (MOMA comunitario) sobre steadycom.py (archivo presentado en el taller 7,
y obtenido del paper original de SteadyCom citado en el reporte).

Etapa i/ii : referencia() -> SteadyCom de un grupo + pFBA (min_norm) -> v*_i = V_i / X_i   
Etapa iii  : techo()      -> SteadyCom de los 3 -> mu_techo
Etapa iv   : ProblemaInercia -> QP: min sum_i w_i ||V_i - X_i v*_i||^2                       (ec. 11)
                                 s.a. SteadyCom a mu = alpha*mu_techo (ec. 12), X_inv >= eps (ec. 13)

Este código se hizo con ayuda de Claude AI, pero con revisión completa y modificaciones significativas por 
parte del grupo, con tal de que se cumpla el objetivo planteado en el informe.
"""

from __future__ import annotations
import cobra
import gurobipy as gp
from gurobipy import GRB
from steadycom import (SteadyCom, SteadyComParams, SteadyComProblem, Status, build_steadycom_model)
import os, sys, contextlib


# ----------------- ETAPAS i y ii --------------------
def referencia(modelos: dict, medio: dict, params=None, x_min=1e-6, **kw):
    """SteadyCom de un grupo (1 modelo = FBA) + pFBA. Devuelve v*_i específicos
    por reacción, con los ids ORIGINALES de cada modelo."""
    com = build_steadycom_model(list(modelos.values()), list(modelos.keys()), **kw)
    sol = SteadyCom(com, medium=medio, params=params, close_other_exchanges=True)
    if sol.status != Status.OPTIMAL or sol.stat != "optimal":
        raise RuntimeError(f"Referencia {list(modelos)}: stat={sol.stat}. "
                           f"Revisar medio (¿puede crecer sin socios?).")
    v = {}
    for org in modelos:
        X = sol.abundance[org]
        if X < x_min:
            raise RuntimeError(f"{org} tiene X≈0 en la referencia: no es un residente real.")
        v[org] = {rid: float(f) / X for rid, f in sol.organism_fluxes(org).items()}
    return dict(mu=sol.growth_rate, X=dict(sol.abundance), v=v, sol=sol, community=com)


# ----------------- ETAPA iii --------------------
def techo(modelos, medio, params=None, minimos=(), eps=1e-4, **kw):
    """mu máximo de la comunidad. Con `minimos`, exige X_i >= eps para esas especies."""
    params = params or SteadyComParams()
    com = build_steadycom_model(list(modelos.values()), list(modelos.keys()), **kw)
    if minimos:
        bc = [({o: 1.0}, ">", eps) for o in minimos]
        prob = SteadyComProblem(com, params=params, biomass_constraints=bc)
        prob.set_medium(medio, close_others=True)
        mu, stat, _ = prob.max_growth_rate()
    else:
        sol = SteadyCom(com, medium=medio, params=params, close_other_exchanges=True)
        mu, stat = sol.growth_rate, sol.stat
    if stat != "optimal":
        raise RuntimeError(f"techo: stat={stat}")
    return mu, com


# ----------------- ETAPA iv -----------------------
# Aquí está la novedad/reimplementación de herramientas existentes
class ProblemaInercia:
    """
    Resolver las ecuaciones 2 a 6 no depende de alpha, por lo que se resuleven una vez.
    Luego, se construye el objetivo modificado para cumplir con las ecuaciones 12 a 14 y la 
    función objetivo (ec. 11). 
    """

    # Sin dependencia de alpha
    def __init__(self, community, medio, refs, invasores, pesos="uno",
                 eps=1e-4, params=None, todos_presentes=True):
        self.com = community
        self.params = params or SteadyComParams()
        self.eps = eps
        self.presentes = list(community.organisms) if todos_presentes else list(invasores)
        self.prob = SteadyComProblem(community, params=self.params)
        self.prob.set_medium(medio, close_others=True)
        self._construir_objetivo(refs, pesos)

    def _construir_objetivo(self, refs, pesos):
        p = self.prob
        obj = gp.QuadExpr()
        self.pesos = {}
        for org in self.com.organisms:
            vref = refs[org]
            w = 1.0
            if pesos == "norma":                      # w_i = 1/||v*_i||^2
                w = 1.0 / max(sum(x * x for x in vref.values()), 1e-12)
            self.pesos[org] = w
            X = p.X[org]
            for rid, vstar in vref.items():
                merged = self.com.reaction_map.get((org, rid))
                if merged is None:
                    continue
                e = p.v[merged] - float(vstar) * X    # V_ij - X_i v*_ij   
                obj.add(e * e, w)                     # ec. 11
        self._obj = obj

    def resolver(self, mu):
        p, gm = self.prob, self.prob.gp_model
        p.set_growth_rate(mu)
        p.set_total_biomass_window(self.params.bm_weight, self.params.bm_weight)
        for org in self.presentes:
            p.X[org].LB = self.eps
        gm.setObjective(self._obj, GRB.MINIMIZE)
        _optimizar_robusto(gm)          # antes: gm.optimize()
        g = gm.Status
        if g == GRB.OPTIMAL:
            X = p.read_biomass(); tot = sum(X.values())
            out = dict(estado="ok", costo=gm.ObjVal, X={k: v / tot for k, v in X.items()})
        elif g in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):   # objetivo >= 0: no puede ser no acotado
            out = dict(estado="inviable", costo=None, X=None)
        else:                                          # 12 numérico, 13 subóptimo, etc.
            out = dict(estado="error", costo=None, X=None)
        out.update(mu=mu, gstat=int(g))
        for org in self.presentes:
            p.X[org].LB = 0.0
        p.set_total_biomass_window(None, None)
        return out


# ----------------- SALIDAS Y MÉTRICAS -------------------------------
def PE(X1, X2):
    # efecto de la proridad (ec. 16)
    return 0.5 * sum(abs(X1[o] - X2[o]) for o in X1)

def delta_residentes(X, Xref, residentes):
    s = sum(X[o] for o in residentes) # (ec. 15)
    if s <= 1e-12:
        return 1.0
    return 0.5 * sum(abs(X[o] / s - Xref[o]) for o in residentes)


# ---------------- PROGRAMA PRINCIPAL -----------------------
def _refs_orden(B, residentes, invasores, medio, medios_ref, params, **kw):
    ref_res = referencia({k: B[k] for k in residentes}, medio, params, **kw)
    refs = dict(ref_res["v"])
    for k in invasores:                               # cada invasor, en solitario
        refs.update(referencia({k: B[k]}, medios_ref.get(k, medio), params, **kw)["v"])
    return refs, ref_res["X"]


def barrido_prioridad(B, medio, alphas, orden1=(("B1", "B2"), ("B3",)),
                      medios_ref=None, eps=1e-4, pesos="uno", params=None,
                      coexistencia=True, **kw):
    params = params or SteadyComParams()
    medios_ref = medios_ref or {}
    res1, inv1 = orden1
    res2, inv2 = inv1, res1

    mu_t, com = techo(B, medio, params, minimos=list(B) if coexistencia else (),
                      eps=eps, **kw)                                   # etapa iii
    refs1, Xref1 = _refs_orden(B, res1, inv1, medio, medios_ref, params, **kw)
    refs2, Xref2 = _refs_orden(B, res2, inv2, medio, medios_ref, params, **kw)
    P1 = ProblemaInercia(com, medio, refs1, inv1, pesos, eps, params, coexistencia)
    P2 = ProblemaInercia(com, medio, refs2, inv2, pesos, eps, params, coexistencia)

    filas = []
    for a in alphas:
        r1, r2 = P1.resolver(a * mu_t), P2.resolver(a * mu_t)
        f = dict(alpha=a, orden1=r1, orden2=r2, PE=None, d1=None, d2=None)
        if r1["estado"] == "ok" and r2["estado"] == "ok":
            f["PE"] = PE(r1["X"], r2["X"])
            f["d1"] = delta_residentes(r1["X"], Xref1, res1)
            f["d2"] = delta_residentes(r2["X"], Xref2, res2)
        filas.append(f)
    return dict(mu_techo=mu_t, filas=filas)


# Funciones auxiliares para silenciar y ordenar salida de la consola
@contextlib.contextmanager
def silencio():
    """Silencia stdout/stderr a nivel de descriptor (captura la salida de Gurobi)."""
    sys.stdout.flush(); sys.stderr.flush()
    nulo = os.open(os.devnull, os.O_WRONLY)
    g1, g2 = os.dup(1), os.dup(2)
    os.dup2(nulo, 1); os.dup2(nulo, 2)
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(g1, 1); os.dup2(g2, 2)
        for fd in (nulo, g1, g2):
            os.close(fd)

def tabla(res, nd=3):
    """Imprime solo lo importante del barrido."""
    print(f"mu_techo = {res['mu_techo']:.4f} 1/h")
    print(f"{'alpha':>6} | {'orden 1':<34} | {'orden 2':<34} | {'PE':>5}")
    def celda(r):
        if r["estado"] == "ok":
            return " ".join(f"{k}={v:.{nd}f}" for k, v in r["X"].items())
        return r["estado"] + (f" (gurobi {r['gstat']})" if r["estado"] == "error" else "")
    for f in res["filas"]:
        pe = "-" if f["PE"] is None else f"{f['PE']:.3f}"
        print(f"{f['alpha']:>6.3f} | {celda(f['orden1']):<34} | {celda(f['orden2']):<34} | {pe:>5}")

# Funcion auxiliar para optimizar de manera robusta
def _optimizar_robusto(gm):
    """Reintenta con ajustes numéricos crecientes hasta obtener un estado definitivo."""
    ajustes = [
        {},
        dict(FeasibilityTol=1e-6, OptimalityTol=1e-6),
        dict(NumericFocus=3, BarHomogeneous=1),
        dict(Method=0, NumericFocus=3),          # simplex primal como último recurso
    ]
    for aj in ajustes:
        for k, v in aj.items():
            gm.setParam(k, v)
        gm.optimize()
        if gm.Status in (GRB.OPTIMAL, GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            break
    return gm.Status