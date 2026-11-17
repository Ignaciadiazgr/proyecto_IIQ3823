
"""
Modelo comunitario con referencias metabólicas individuales pFBA y referencia de abundancias 
comunitaria SteadyCom.

Etapas:
1. Recibir las referencias calculadas en metodo/referencias.py.
2. Construir el modelo comunitario de estado estacionario.
3. Construir el problema MOMA con referencias pFBA individuales.
4. Evaluar las abundancias para distintos crecimientos comunitarios.
5. Comparar los resultados entre órdenes de llegada.

Se conserva la formulación MOMA original:
    min sum_i w_i * sum_j (V_ij - X_i * v*_ij)^2

Sujeto a:
    S V = 0
    X_i * lb_ij <= V_ij <= X_i * ub_ij
    V_biomasa_i = mu * X_i
    sum_i X_i = 1

Pesos:
    w_i = 1 / ||v*_i||²   ("norma")
    w_i = 1               ("uno")
"""

from __future__ import annotations
import math
import contextlib
import io
import gurobipy as gp
from gurobipy import GRB
from metodo.steadycom_adapted import SteadyCom


# parámetros
COTA = 100
PESOS_VALIDOS = ("norma", "uno")

# ============================================================
# Preparación de datos 
# ============================================================
def constraints_medio(com, medio):
    """
    Convierte el medio de cultivo en cotas para los intercambios del modelo comunitario.

    Convención BiGG:
        flujo negativo: captación
        flujo positivo: secreción

    Los metabolitos no incluidos en el medio no pueden captarse, pero sí pueden secretarse.

    Parámetros
    ----------
    com : reframed.Community
        Comunidad metabólica.
    medio : dict
        {metabolito: (lb, ub)}

    Retorna
    -------
    dict
        {reacción_exchange: (lb, ub)}
    """

    merged = com.merged_model
    restricciones = {}
    utilizados = set()

    for rid in merged.get_exchange_reactions():
        metabolito = next(iter(merged.reactions[rid].stoichiometry))

        # Eliminar prefijo M_ utilizado por ReFramed.
        base = (metabolito[2:] if metabolito.startswith("M_") else metabolito)
        lb, ub = medio.get(base, (0.0, COTA))
        restricciones[rid] = (max(float(lb), -COTA), min(float(ub), COTA))
        utilizados.add(base)

    # para robustez
    faltantes = set(medio) - utilizados
    if faltantes:
        raise ValueError("Metabolitos del medio ausentes en la comunidad: " f"{sorted(faltantes)}")

    return restricciones


def _modelo_base(com, medio):
    """
    Construye el modelo comunitario:
        S V = 0 (E.E.)
        sum_i X_i = 1 (suma abundancias)
        X_i * lb_ij <= V_ij <= X_i * ub_ij (flujos)

    No incorpora todavía las restricciones de crecimiento
    ni una función objetivo.

    Retorna
    -------
    gm : gurobipy.Model
    X  : variables de abundancia.
    V  : variables de flujo comunitario.
    """

    merged = com.merged_model
    restricciones_medio = constraints_medio(com, medio)
    intercambios = set(merged.get_exchange_reactions())
    gm = gp.Model("Community_MOMA")
    gm.Params.OutputFlag = 0

    # abundancias
    X = {
        org: gm.addVar(
            lb=0.0,
            ub=1.0,
            name=f"X_{org}"
        )
        for org in com.organisms
    }

    # flujos
    V = {}
    for rid, reaccion in merged.reactions.items():
        if rid in intercambios:
            lb, ub = restricciones_medio[rid]
        else:
            lb = (-GRB.INFINITY if reaccion.lb < 0 else 0.0)
            ub = (GRB.INFINITY if reaccion.ub > 0 else 0.0)

        V[rid] = gm.addVar(lb=lb, ub=ub, name=f"V_{rid}")

    gm.update()

    # suma abundancias = 1
    gm.addConstr(gp.quicksum(X.values()) == 1.0, name="abundancia_total")

    # Estado estacionario: S V = 0
    for metabolito, fila in merged.metabolite_reaction_lookup().items():
        gm.addConstr(gp.quicksum(float(coeficiente) * V[rid] for rid, coeficiente in fila.items()) == 0, name=f"balance_{metabolito}")

    # cotas de flujo
    for org, organismo in com.organisms.items():
        for rid, reaccion in organismo.reactions.items():
            mid = com.reaction_map.get((org, rid))
            if mid is None:
                continue

            # La reacción de biomasa se restringe por separado.
            if rid == organismo.biomass_reaction:
                continue
            lb = (-COTA if math.isinf(reaccion.lb) else float(reaccion.lb))
            ub = (COTA if math.isinf(reaccion.ub) else float(reaccion.ub))

            if lb != 0:
                gm.addConstr(V[mid] >= lb * X[org])

            if ub != 0:
                gm.addConstr(V[mid] <= ub * X[org])

    gm.update()

    return gm, X, V


def _fijar_crecimiento(gm, com, X, V, mu):
    """
    Impone crecimiento balanceado:
        V_biomasa_i = mu * X_i
    para cada organismo.
    """

    restricciones = []
    for org, organismo in com.organisms.items():
        rid_biomasa = organismo.biomass_reaction
        mid = com.reaction_map.get((org, rid_biomasa))
        # para robustez
        if mid is None:
            raise ValueError(f"No se encontró la reacción de biomasa de {org}.")
        restricciones.append(gm.addConstr(V[mid] == float(mu) * X[org], name=f"crecimiento_{org}"))

    return restricciones


# Resolución del problema
def _optimizar(gm, tiempo_limite=600, mip_gap=None, mostrar_log=False):
    """
    Resuelve el modelo no convexo buscando un óptimo local.
    """

    gm.Params.OptimalityTarget = 1
    gm.Params.NonConvex = -1
    gm.Params.TimeLimit = tiempo_limite
    gm.Params.OutputFlag = 1 if mostrar_log else 0
    gm.optimize()

    return gm.Status


def comprobar_base(referencias, medio, abs_tol=1e-5):
    """
    Comprueba si el modelo comunitario reconstruido reproduce la factibilidad de la referencia SteadyCom.

    Comprueba:
        - Factibilidad ligeramente por debajo de mu máximo.
        - Infactibilidad ligeramente por encima de mu máximo.

    Retorna
    -------
    tuple(bool, bool)
    """

    com = referencias["steadycom"]["comunidad"]
    mu = float(referencias["steadycom"]["mu_max"])

    def factible(mu_prueba):
        gm, X, V = _modelo_base(com, medio)
        _fijar_crecimiento(gm, com, X, V, mu_prueba)
        estado = _optimizar(gm)
        gm.dispose()
        return estado == GRB.OPTIMAL

    inferior = factible(mu * (1 - 1e-6))
    superior = factible(mu + 5 * abs_tol)

    return inferior, superior


# ============================================================
# REFERENCIAS
# ============================================================
def preparar_referencias_moma(referencias, com):
    referencias_pfba = referencias["pfba"]
    refs = {}
    for org in com.organisms:
        if org not in referencias_pfba:
            raise ValueError(f"No existe una referencia pFBA para {org}.")
        flujos = referencias_pfba[org]["flujos"]
        refs_org = {}

        for rid, flujo in flujos.items():
            # Obtener identificador comunitario
            mid = com.reaction_map.get((org, rid))
            if mid is None:
                continue
            # Guardar utilizando el identificador comunitario
            refs_org[mid] = float(flujo)

        if not refs_org:
            raise ValueError(f"No se encontraron flujos compatibles para {org}.")

        refs[org] = refs_org

    return refs


def referencia_abundancias_residentes(abundancias, residentes):
    """
    Normaliza las abundancias SteadyCom considerando únicamente los organismos residentes.
    La normalización permite comparar la composición relativa de los residentes antes y después de la invasión.
    """

    total = sum(abundancias[org] for org in residentes)

    # para robustez
    if total <= 1e-12:
        raise ValueError("La abundancia total de los residentes es cero.")

    return {org: abundancias[org] / total for org in residentes}


def calcular_techo(referencias, medio, coexistencia=True, eps=1e-4, abs_tol=1e-5, max_iters=60):
    """
    Obtiene el crecimiento máximo que se utilizará en el barrido.

    Si coexistencia=True:
        exige X_i >= eps para todos los organismos.

    Si coexistencia=False:
        utiliza el crecimiento máximo SteadyCom sin restricciones adicionales de abundancia.
    """

    ref_com = referencias["steadycom"]
    com = ref_com["comunidad"]
    if not coexistencia:
        return float(ref_com["mu_max"])

    restricciones = constraints_medio(com, medio)
    for org in com.organisms:
        restricciones[f"x_{org}"] = (float(eps), 1.0)

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        solucion = SteadyCom(com, constraints=restricciones, abs_tol=abs_tol, max_iters=max_iters, verbose=False)

    return float(solucion.growth)


# Problema MOMA
class ProblemaInercia:
    """
    Construye el problema de inercia con la función objetivo:
        Z = sum_i w_i ||X_i*v_i - X_i_ref*v_i_ref||^2 + sum_i w_i ||v_i - v_i_ref||^2

    Parámetros
    ----------
    com : Community de ReFramed (comunidad metabólica).
    medio : dict con el medio de cultivo comunitario.
    refs : flujos de referencia por organismo, refs[org][id_rxn_comunitaria] = flujo.
    X_ref : abundancias de referencia, X_ref[org] = abundancia.
    invasores : lista de organismos que ingresan en la comunidad.
    pesos : "norma" (w_i = 1 / ||v_i_ref||^2) o "uno" (w_i = 1).
    eps : abundancia mínima para organismos que deben estar presentes.
    todos_presentes : True exige a todos los organismos, False solo a los invasores.
    """

    def __init__(self, com, medio, refs, X_ref, invasores, pesos="norma", eps=1e-4,
                 todos_presentes=True):
        self.restricciones_crecimiento = []
        self.com, self.eps, self.pesos_tipo = com, eps, pesos
        self.invasores = list(invasores)
        self.organismos = list(com.organisms.keys())

        # Validación de referencias
        faltantes = set(self.organismos) - set(refs.keys())
        if faltantes:
            raise ValueError(f"Faltan referencias de flujo para: {faltantes}")
        
        faltantes_X = set(self.organismos) - set(X_ref.keys())
        if faltantes_X:
            raise ValueError(f"Faltan abundancias de referencia para: {faltantes_X}")
        
        if pesos not in ("norma", "uno"):
            raise ValueError("El parámetro pesos debe ser 'norma' o 'uno'.")
        
        if any(X_ref[org] < 0 for org in self.organismos):
            raise ValueError("Las abundancias de referencia no pueden ser negativas.")
        
        if not any(X_ref[org] > 0 for org in self.organismos):
            raise ValueError("Las abundancias de referencia no pueden ser todas cero.")

        self.X_ref = {org: float(X_ref[org]) for org in self.organismos}
        self.refs = refs
        self.presentes = self.organismos.copy() if todos_presentes else self.invasores.copy()
        # Modelo base
        self.gm, self.X, self.V = _modelo_base(com, medio)
        self.gm.Params.NonConvex = 2
        self.gm.Params.OutputFlag = 0

        # Flujos específicos: (organismo, reacción comunitaria) -> variable / reacción local
        self.v, self.reacciones = {}, {}
        for org, modelo in com.organisms.items():
            for rid, reaccion in modelo.reactions.items():
                # Obtener el identificador de la reacción comunitaria
                mid = com.reaction_map.get((org, rid))
                # Algunas reacciones locales no tienen mapeo comunitario
                if mid is None:
                    continue
                # Considerar únicamente reacciones con referencia pFBA
                if mid not in refs[org]:
                    continue
                self.reacciones[(org, mid)] = reaccion
                self.v[(org, mid)] = self.gm.addVar(lb=reaccion.lb, ub=reaccion.ub, name=f"v_{org}_{mid}")
        self.gm.update()

        # Restricciones de presencia
        for org in self.presentes:
            self.gm.addConstr(self.X[org] >= eps, name=f"presencia_{org}")

        # Vinculación entre flujos comunitarios y específicos: V_ij = X_i * v_ij
        for (org, mid), variable_v in self.v.items():
            self.gm.addQConstr(self.V[mid] == self.X[org] * variable_v, name=f"relacion_flujos_{org}_{mid}")

        # Ponderaciones
        self.w = {}
        for org in self.organismos:
            if pesos == "norma":
                norma2 = sum(float(valor) ** 2 for valor in refs[org].values())
                if norma2 <= 1e-12:
                    raise ValueError(
                        f"La norma de los flujos de referencia de {org} es cero o muy pequeña.")
                self.w[org] = 1.0 / norma2
            else:
                self.w[org] = 1.0

        # Función objetivo por partes
        obj_com = gp.quicksum(self.w[org] * (self.V[mid] - self.X_ref[org] * float(refs[org][mid])) ** 2 for (org, mid) in self.v.keys())
        obj_esp = gp.quicksum(self.w[org] * (self.v[(org, mid)] - float(refs[org][mid])) ** 2 for (org, mid) in self.v.keys())
        self.gm.setObjective(obj_com + obj_esp, GRB.MINIMIZE)
        self.gm.update()

    
    def resolver(self, mu, tiempo_limite=600):
        """Resuelve el problema para una tasa de crecimiento mu."""

        # Eliminar las restricciones de crecimiento anteriores
        if self.restricciones_crecimiento:
            self.gm.remove(self.restricciones_crecimiento)
            self.gm.update()

        # Incorporar el nuevo crecimiento
        self.restricciones_crecimiento = _fijar_crecimiento(self.gm, self.com, self.X, self.V, mu)
        estado = _optimizar(self.gm, tiempo_limite=tiempo_limite)

        nombres_estados = {
            GRB.OPTIMAL: "Óptimo global",
            GRB.SUBOPTIMAL: "Subóptimo",
            GRB.TIME_LIMIT: "Límite de tiempo",
            GRB.INFEASIBLE: "Infactible",
            GRB.INF_OR_UNBD: "Infactible o no acotado",
            GRB.UNBOUNDED: "No acotado",
        }

        if hasattr(GRB, "LOCALLY_OPTIMAL"):
            nombres_estados[GRB.LOCALLY_OPTIMAL] = "Óptimo local"

        estado_local = getattr(GRB, "LOCALLY_OPTIMAL", None)

        if estado_local is not None:
            nombres_estados[estado_local] = "Óptimo local"

        resultado = {
            "mu": float(mu),
            "status": estado,
            "status_texto": nombres_estados.get(estado, "Otro estado"),
            "solucion_factible": False,
            "objetivo": None,
            "gap": None,
            "tiempo": float(self.gm.Runtime),
            "abundancias": None,
            "flujos_comunitarios": None,
            "flujos_especificos": None,
            "costo_comunidad": None,
            "costo_especifico": None,
            "pesos": self.w.copy()
        }

        # Si no hay ninguna solución factible, retornar
        if self.gm.SolCount == 0:
            return resultado

        # Hay una solución factible, aunque no necesariamente óptima
        resultado["solucion_factible"] = True
        resultado["objetivo"] = float(self.gm.ObjVal)

        try:
            resultado["gap"] = float(self.gm.MIPGap)
        except (AttributeError, gp.GurobiError):
            resultado["gap"] = None

        # Extraer valores de las variables
        X_sol = {
            org: float(self.X[org].X)
            for org in self.organismos
        }

        V_sol = {
            mid: float(variable.X)
            for mid, variable in self.V.items()
        }

        v_sol = {
            k: float(variable.X)
            for k, variable in self.v.items()
        }

        # Evaluar separadamente los dos términos del objetivo
        costo_comunidad = 0.0
        costo_especifico = 0.0

        for (org, mid) in self.v.keys():
            v_ref = float(self.refs[org][mid])

            costo_comunidad += (
                self.w[org]
                * (V_sol[mid] - self.X_ref[org] * v_ref) ** 2
            )

            costo_especifico += (
                self.w[org]
                * (v_sol[(org, mid)] - v_ref) ** 2
            )

        resultado.update({
            "abundancias": X_sol,
            "flujos_comunitarios": V_sol,
            "flujos_especificos": v_sol,
            "costo_comunidad": float(costo_comunidad),
            "costo_especifico": float(costo_especifico)
        })

        return resultado


    def cerrar(self):
        """Libera el modelo de Gurobi."""
        self.gm.dispose()



# Barrido MOMA sin efecto de prioridad
def barrido_moma(referencias, medio, alphas, eps=1e-4, pesos="norma", coexistencia=True, abs_tol=1e-5, max_iters=60):
    """
    Evalúa el modelo MOMA para distintos valores de crecimiento comunitario y compara los resultados con SteadyCom.

    Utiliza:
        - Flujos individuales pFBA como referencias metabólicas.
        - Abundancias SteadyCom como referencia comunitaria.
        - Un único modelo MOMA para todo el barrido.

    Retorna:
        Diccionario con referencias, valores de mu y resultados MOMA.
    """

    # Referencias SteadyCom
    ref_com = referencias["steadycom"]
    com = ref_com["comunidad"]

    mu_steadycom = float(ref_com["mu_max"])
    abundancias_steadycom = ref_com["abundancias"]

    # Referencias metabólicas individuales pFBA
    refs = preparar_referencias_moma(referencias, com)

    # Máximo crecimiento bajo las condiciones de coexistencia
    mu_techo = calcular_techo(referencias, medio, coexistencia, eps, abs_tol, max_iters)
    if mu_techo <= 0:
        raise RuntimeError("El crecimiento máximo comunitario no es positivo.")

    # Construir un único problema MOMA
    problema = ProblemaInercia(
        com=com,
        medio=medio,
        refs=refs,
        X_ref=abundancias_steadycom,
        invasores=[],
        pesos=pesos,
        eps=eps,
        todos_presentes=coexistencia
    )

    filas = []
    try:
        for alpha in alphas:
            alpha = float(alpha)
            mu = alpha * mu_techo
            resultado = problema.resolver(mu)
            filas.append({
                "alpha": alpha,
                "mu": mu,
                "resultado": resultado
            })

    finally:
        problema.cerrar()

    # Mayor crecimiento factible encontrado en el barrido.
    soluciones_factibles = [
        fila["mu"]
        for fila in filas
        if fila["resultado"]["solucion_factible"]
    ]

    mu_moma_max_barrido = (
        max(soluciones_factibles)
        if soluciones_factibles
        else None
    )

    return {
        "mu_steadycom": mu_steadycom,
        "mu_techo": mu_techo,
        "mu_moma_max_barrido": mu_moma_max_barrido,
        "abundancias_steadycom": abundancias_steadycom,
        "filas": filas,
        "community": com,
        "referencias": referencias,
        "refs_moma": refs,
        "pesos": {
            "tipo": pesos,
            "valores": problema.w.copy()
        }
    }





# ============================================================
# Métricas para implementación futura
# ============================================================
def PE(X1, X2):
    """
    Efecto de prioridad (PE, priority effect) entre dos composiciones de abundancia.
    Distancia de variación total entre dos composiciones.
    PE = 0.5 * sum_i |X1_i - X2_i|
    """
    return 0.5 * sum(abs(X1[org] - X2[org]) for org in X1)


def delta_residentes(X, Xref, residentes):
    """
    Cuantifica el cambio en la composición relativa de los residentes respecto de su referencia.
    """
    total = sum(X[org]for org in residentes) 
    return 0.5 * sum(abs(X[org] / total - Xref[org]) for org in residentes)



def barrido_prioridad(referencias, medio, X_ref, alphas,  orden1=(("B1", "B2"), ("B3",)), eps=1e-4,
                      pesos="norma", coexistencia=True, abs_tol=1e-5, max_iters=60):
    """
    Compara dos órdenes de llegada mediante MOMA.

    Parámetros
    ----------
    referencias : dict, resultado de calcular_referencias().
    medio : dict, medio compartido.
    alphas : iterable, factores aplicados al crecimiento máximo.
    orden1 : tuple, (residentes iniciales, invasores).
    eps : abundancia mínima exigida.
    pesos : "norma" o "uno".
    coexistencia : si es True, exige presencia de todos los organismos.

    Retorna
    -------
    dict con los resultados para cada alpha y orden de llegada.
    """
    # 1. Recuperar referencias
    ref_com = referencias["steadycom"]
    com = ref_com["comunidad"]
    abundancias_steadycom = ref_com["abundancias"]
    refs = preparar_referencias_moma(referencias, com)

    # 2. Crecimiento máximo de referencia
    mu_techo = calcular_techo(referencias=referencias, medio=medio, coexistencia=coexistencia,
                              eps=eps, abs_tol=abs_tol, max_iters=max_iters)
    if mu_techo <= 0:
        raise RuntimeError("El crecimiento máximo comunitario no es positivo.")

    # 3. Definición de órdenes de llegada
    residentes1, invasores1 = orden1
    residentes2, invasores2 = invasores1, residentes1

    # Verificar que los organismos existan
    organismos = set(com.organisms)
    if set(residentes1 + invasores1) != organismos:
        raise ValueError(
            "El primer orden debe incluir todos los organismos exactamente una vez.")
    if set(residentes2 + invasores2) != organismos:
        raise ValueError(
            "El segundo orden debe incluir todos los organismos exactamente una vez.")

    # 4. Referencias de abundancia de residentes
    Xref1 = referencia_abundancias_residentes(abundancias_steadycom, residentes1)
    Xref2 = referencia_abundancias_residentes(abundancias_steadycom, residentes2)

    print("\nREFERENCIAS DE ABUNDANCIA")
    print("Orden 1:", Xref1)
    print("Orden 2:", Xref2)

    print("\nINVASORES")
    print("Orden 1:", invasores1)
    print("Orden 2:", invasores2)

    # 5. Construir problemas MOMA
    P1 = ProblemaInercia(com=com, medio=medio, refs=refs, X_ref=X_ref, invasores=invasores1, pesos=pesos,
                         eps=eps, todos_presentes=coexistencia)
    P2 = ProblemaInercia(com=com, medio=medio, refs=refs, X_ref=X_ref, invasores=invasores2, pesos=pesos,
                         eps=eps, todos_presentes=coexistencia)

    print("\nREFERENCIAS INTERNAS")
    print("P1:", P1.X_ref)
    print("P2:", P2.X_ref)

    print("\nORGANISMOS PRESENTES")
    print("P1:", P1.presentes)
    print("P2:", P2.presentes)

    # 6. Barrido de crecimiento
    filas = []
    try:
        for alpha in alphas:
            mu = float(alpha) * mu_techo
            r1 = P1.resolver(mu)
            r2 = P2.resolver(mu)

            fila = {
                "alpha": float(alpha),
                "mu": mu,
                "orden1": r1,
                "orden2": r2,
                "PE": None,
                "d1": None,
                "d2": None
            }

            # Calcular métricas solo si ambos órdenes tienen
            # soluciones factibles.
            if r1["solucion_factible"] and r2["solucion_factible"]:
                fila.update({
                    "PE": PE(r1["abundancias"], r2["abundancias"]),
                    "d1": delta_residentes(
                        r1["abundancias"], Xref1, residentes1
                    ),
                    "d2": delta_residentes(
                        r2["abundancias"], Xref2, residentes2
                    )
                })

            filas.append(fila)

    finally:
        P1.cerrar()
        P2.cerrar()

    return {"mu_techo": mu_techo, "mu_steadycom": float(ref_com["mu_max"]), "filas": filas,
            "community": com, "referencias": referencias, "refs_moma": refs,
            "pesos": {"tipo": pesos, "orden1": P1.w, "orden2": P2.w}}