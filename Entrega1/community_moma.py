"""
SteadyCom con inercia metabolica sobre reframed.

Etapa i/ii : referencia()  -> SteadyCom adaptado de reframed (steadycom_adapted) + pFBA a composicion fija -> v*_i = V_i/X_i
Etapa iii  : techo()       -> SteadyCom adaptado de reframed de todos los miembros -> mu_techo
Etapa iv   : ProblemaInercia -> QP: min sum_i w_i ||V_i - X_i v*_i||^2        (ec. 11)
                                con w_i = 1/||v*_i||^2                         (ec. 14)
                                s.a. S V=0, X_i l <= V <= X_i u, V_bio = mu X_i, sum X=1   (ecs. 2-5)
                                con mu = alpha*mu_techo                        (ec. 12)
                                y X_i >= eps                                   (ec. 13)
Validacion : comprobar_base() contrasta el SteadyCom reconstruido con el oficial.
"""
from __future__ import annotations
import math
import cobra
import gurobipy as gp
from gurobipy import GRB
from reframed import from_cobrapy
from reframed.community.model import Community
from reframed.core.model import ReactionType
from steadycom_adapted import SteadyCom
from utils import tabla, silencio

COTA = 1000    # cota de intercambios no listados / secrecion y flujos internos no acotados

# --------comunidad
def a_reframed(modelo: cobra.Model, org_id: str):
    """cobra.Model a CBModel de reframed MARCANDO el compartimento externo y los intercambios.
    Sin esto Community.merge_models no crea pool compartido."""
    ids_ex = {r.id for r in modelo.exchanges}
    rf = from_cobrapy(modelo.copy())
    rf.id = org_id
    externos = set()
    for rid in ids_ex:
        if rid not in rf.reactions:
            raise ValueError(f"{org_id}: el intercambio {rid} no existe tras from_cobrapy")
        rxn = rf.reactions[rid]
        rxn.reaction_type = ReactionType.EXCHANGE
        for m_id in rxn.stoichiometry:
            externos.add(rf.metabolites[m_id].compartment)
    for c_id in externos:
        rf.compartments[c_id].external = True
    return rf


def crear_comunidad(modelos: dict, **kw):
    com = Community("comunidad", [a_reframed(m, org) for org, m in modelos.items()], **kw)
    if not com.merged_model.get_exchange_reactions():
        raise RuntimeError("El modelo fusionado no tiene intercambios comunitarios (R_EX_*): "
                           "revisar compartimento externo y reaction_type.")
    return com


def constraints_medio(com, medio: dict):
    """Medio -> cotas de los R_EX_* comunitarios. Convencion BiGG: flujo<0 = captacion.
    No listado: sin captacion, secrecion libre. Falla si el medio no casa con la comunidad."""
    merged = com.merged_model
    cons, usados = {}, set()
    for rid in merged.get_exchange_reactions():
        met = next(iter(merged.reactions[rid].stoichiometry))
        base = met[2:] if met.startswith("M_") else met
        lb, ub = medio.get(base, (0.0, COTA))
        cons[rid] = (max(float(lb), -COTA), min(float(ub), COTA))
        usados.add(base)
    faltan = set(medio) - usados
    if faltan:
        raise ValueError(f"Metabolitos del medio ausentes en la comunidad: {sorted(faltan)}")
    return cons


# --------------------- SteadyCom reconstruido (extension)
def _modelo_base(com, medio):
    """S V=0, X_i l <= V <= X_i u, sum X=1, cotas del medio. Sin crecimiento ni objetivo."""
    merged = com.merged_model
    cons = constraints_medio(com, medio)
    exch = set(merged.get_exchange_reactions())
    gm = gp.Model("base"); gm.Params.OutputFlag = 0

    X = {o: gm.addVar(lb=0.0, ub=1.0, name=f"X_{o}") for o in com.organisms}
    V = {}
    for rid, rxn in merged.reactions.items():
        if rid in exch:
            lb, ub = cons[rid]
        else:
            lb = -GRB.INFINITY if rxn.lb < 0 else 0.0
            ub = GRB.INFINITY if rxn.ub > 0 else 0.0
        V[rid] = gm.addVar(lb=lb, ub=ub, name=f"V_{rid}")
    gm.update()
    gm.addConstr(gp.quicksum(X.values()) == 1.0, name="abundancia")

    for m_id, fila in merged.metabolite_reaction_lookup().items():
        gm.addConstr(gp.quicksum(float(c) * V[r] for r, c in fila.items()) == 0, name=f"mb_{m_id}")

    for org, organism in com.organisms.items():
        for rid, rxn in organism.reactions.items():
            mid = com.reaction_map.get((org, rid))
            if mid is None or rid == organism.biomass_reaction:
                continue
            lb = -COTA if math.isinf(rxn.lb) else float(rxn.lb)
            ub = COTA if math.isinf(rxn.ub) else float(rxn.ub)
            if lb != 0:
                gm.addConstr(V[mid] >= lb * X[org])
            if ub != 0:
                gm.addConstr(V[mid] <= ub * X[org])
    gm.update()
    return gm, X, V


def _fijar_crecimiento(gm, com, X, V, mu):
    cs = []
    for org, organism in com.organisms.items():
        mid = com.reaction_map[(org, organism.biomass_reaction)]
        cs.append(gm.addConstr(V[mid] == float(mu) * X[org]))
    return cs


def _optimizar(gm):
    """Reintenta con ajustes numericos crecientes hasta un estado definitivo."""
    for aj in ({}, dict(FeasibilityTol=1e-6, OptimalityTol=1e-6),
               dict(NumericFocus=3, BarHomogeneous=1),
               dict(Method=0, NumericFocus=3, Presolve=0)):
        for k, v in aj.items():
            gm.setParam(k, v)
        gm.optimize()
        if gm.Status in (GRB.OPTIMAL, GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            break
    return gm.Status


def comprobar_base(com, medio, mu, abs_tol):
    """Valida el SteadyCom reconstruido contra el oficial: debe ser factible en mu
    y infactible en mu + 5*abs_tol. Devuelve (factible_en_mu, factible_sobre_mu)."""
    def factible(m):
        gm, X, V = _modelo_base(com, medio)
        _fijar_crecimiento(gm, com, X, V, m)
        return _optimizar(gm) == GRB.OPTIMAL
    return factible(mu * (1 - 1e-6)), factible(mu + 5 * abs_tol)



# ---------- ETAPAS 1, 2, 3 ----------
def _resolver(com, medio, minimos=(), eps=1e-4, abs_tol=1e-5, max_iters=60):
    cons = constraints_medio(com, medio)
    for org in minimos:                       # X_i >= eps (limites de las variables x_<org>)
        cons[f"x_{org}"] = (float(eps), 1.0)
    return SteadyCom(com, constraints=cons, abs_tol=abs_tol, max_iters=max_iters)


def referencia(modelos: dict, medio: dict, abs_tol=1e-5, x_min=1e-6, holgura=1e-6, **kw):
    """SteadyCom adaptado + pFBA a composicion fija (X* y mu* del oficial)."""
    com = crear_comunidad(modelos, **kw)
    sol = _resolver(com, medio, abs_tol=abs_tol)
    mu = float(sol.growth)
    X = {o: float(sol.abundance[o]) for o in modelos}
    for o, x in X.items():
        if x < x_min:
            raise RuntimeError(f"{o} tiene X~0 en la referencia (X={x:.2e}): no es un residente real.")

    gm, Xv, V = _modelo_base(com, medio)
    for o, x in X.items():
        Xv[o].LB = Xv[o].UB = x
    _fijar_crecimiento(gm, com, Xv, V, mu * (1 - holgura))
    a = gm.addVars(list(V), lb=0.0)
    for r, v in V.items():
        gm.addConstr(a[r] >= v); gm.addConstr(a[r] >= -v)
    gm.setObjective(gp.quicksum(a.values()), GRB.MINIMIZE)     # pFBA comunitario
    if _optimizar(gm) != GRB.OPTIMAL:
        raise RuntimeError("pFBA a composicion fija infactible: el SteadyCom reconstruido "
                           "no reproduce al oficial (usar comprobar_base).")
    v = {o: {rid: V[com.reaction_map[(o, rid)]].X / X[o]
             for rid in com.organisms[o].reactions if (o, rid) in com.reaction_map}
         for o in modelos}
    return dict(mu=mu, X=X, v=v, sol=sol, community=com)


def techo(modelos: dict, medio: dict, minimos=(), eps=1e-4, abs_tol=1e-5, **kw):
    """mu_techo oficial; con `minimos` exige X_i >= eps (coexistencia)."""
    com = crear_comunidad(modelos, **kw)
    sol = _resolver(com, medio, minimos=minimos, eps=eps, abs_tol=abs_tol)
    return float(sol.growth), com



# ----------- ETAPA 4 -----------
PESOS_VALIDOS = ("norma", "uno")

class ProblemaInercia:
    """Objetivo (11) con pesos de la ec. 14 construido una vez; el barrido solo cambia mu."""
    def __init__(self, com, medio, refs, invasores, pesos="norma", eps=1e-4, todos_presentes=True):
        if pesos not in PESOS_VALIDOS:
            raise ValueError(f"pesos debe ser uno de {PESOS_VALIDOS}")
        self.com, self.eps, self.pesos_tipo = com, float(eps), pesos
        self.presentes = list(com.organisms) if todos_presentes else list(invasores)
        self.gm, self.X, self.V = _modelo_base(com, medio)

        self.w, self.terminos = {}, {}
        obj = gp.QuadExpr()
        for org in com.organisms:
            vref = refs[org]
            norma2 = sum(float(z) ** 2 for z in vref.values())
            if pesos == "norma":
                if norma2 < 1e-12:
                    raise ValueError(f"{org}: ||v*||^2 ~ 0, no se puede normalizar (ec. 14)")
                w = 1.0 / norma2                              # ec. (14)
            else:
                w = 1.0
            self.w[org] = w
            t = gp.QuadExpr()                                 # sum_j (V_ij - X_i v*_ij)^2
            for rid, vstar in vref.items():
                mid = com.reaction_map.get((org, rid))
                if mid is None:
                    continue
                e = self.V[mid] - float(vstar) * self.X[org]
                t.add(e * e)
            self.terminos[org] = t
            obj.add(t, w)                                     # ec. (11)
        self.gm.setObjective(obj, GRB.MINIMIZE)
        self.gm.update()

    def resolver(self, mu):
        gm = self.gm
        cs = _fijar_crecimiento(gm, self.com, self.X, self.V, mu)       # ec. (12) fijando mu
        cs += [gm.addConstr(self.X[o] >= self.eps) for o in self.presentes]   # ec. (13)
        st = _optimizar(gm)
        if st == GRB.OPTIMAL:
            Xr = {o: v.X for o, v in self.X.items()}; t = sum(Xr.values())
            costo_esp = {o: self.w[o] * self.terminos[o].getValue() for o in self.com.organisms}
            out = dict(estado="ok", costo=gm.ObjVal, costo_esp=costo_esp,
                       X={o: x / t for o, x in Xr.items()})
        elif st in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            out = dict(estado="inviable", costo=None, costo_esp=None, X=None)
        else:
            out = dict(estado="error", costo=None, costo_esp=None, X=None)
        out.update(mu=float(mu), gstat=int(st), pesos=self.pesos_tipo)
        gm.remove(cs); gm.update()
        return out



# ---------------- SALIDAS ------------------
def PE(X1, X2):
    return 0.5 * sum(abs(X1[o] - X2[o]) for o in X1)

def delta_residentes(X, Xref, residentes):
    s = sum(X[o] for o in residentes)
    return 1.0 if s <= 1e-12 else 0.5 * sum(abs(X[o] / s - Xref[o]) for o in residentes)

def _refs_orden(B, residentes, invasores, medio, medios_ref, **kw):
    r = referencia({k: B[k] for k in residentes}, medio, **kw)
    refs = dict(r["v"])
    for k in invasores:                                   # cada invasor, en solitario
        refs.update(referencia({k: B[k]}, medios_ref.get(k, medio), **kw)["v"])
    return refs, r["X"]

def barrido_prioridad(B, medio, alphas, orden1=(("B1", "B2"), ("B3",)), medios_ref=None,
                      eps=1e-4, pesos="norma", coexistencia=True, abs_tol=1e-5):
    medios_ref = medios_ref or {}
    res1, inv1 = orden1
    res2, inv2 = inv1, res1
    mu_t, com = techo(B, medio, minimos=list(B) if coexistencia else (), eps=eps, abs_tol=abs_tol)
    refs1, Xref1 = _refs_orden(B, res1, inv1, medio, medios_ref, abs_tol=abs_tol)
    refs2, Xref2 = _refs_orden(B, res2, inv2, medio, medios_ref, abs_tol=abs_tol)
    P1 = ProblemaInercia(com, medio, refs1, inv1, pesos, eps, coexistencia)
    P2 = ProblemaInercia(com, medio, refs2, inv2, pesos, eps, coexistencia)
    filas = []
    for a in alphas:
        r1, r2 = P1.resolver(a * mu_t), P2.resolver(a * mu_t)
        f = dict(alpha=a, orden1=r1, orden2=r2, PE=None, d1=None, d2=None)
        if r1["estado"] == r2["estado"] == "ok":
            f.update(PE=PE(r1["X"], r2["X"]), d1=delta_residentes(r1["X"], Xref1, res1),
                     d2=delta_residentes(r2["X"], Xref2, res2))
        filas.append(f)
    return dict(mu_techo=mu_t, filas=filas, community=com,
                pesos=dict(tipo=pesos, orden1=P1.w, orden2=P2.w))
