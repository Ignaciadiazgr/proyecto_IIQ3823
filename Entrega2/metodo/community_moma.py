from __future__ import annotations

import math
import contextlib
import io
import time
import gurobipy as gp
from gurobipy import GRB
from metodo.referencias import calcular_referencia_comunitaria
from metodo.steadycom_adapted import SteadyCom


# ============================================================
# CONFIGURACIÓN
# ============================================================

COTA = 100
PESOS_VALIDOS = ("norma", "uno")


# ============================================================
# RESTRICCIONES DEL MEDIO
# ============================================================

def constraints_medio(com, medio):

    restricciones = {}
    intercambios = {}

    # Identificar intercambios comunitarios.
    for rxn_id, rxn in com.merged_model.reactions.items():

        if not rxn_id.startswith("R_EX_"):
            continue

        metabolitos = list(rxn.stoichiometry.keys())

        if len(metabolitos) != 1:
            continue

        met_id = metabolitos[0]
        intercambios[met_id] = rxn_id

    # Verificar que todos los metabolitos del medio
    # tengan una reacción de intercambio comunitaria.
    faltantes = set(medio) - set(intercambios)

    if faltantes:
        raise ValueError(
            f"No existen reacciones de intercambio comunitarias "
            f"para los metabolitos: {faltantes}"
        )

    # Definir cotas para todos los intercambios.
    for met_id, rxn_id in intercambios.items():

        if met_id in medio:
            lb, ub = medio[met_id]
        else:
            # Impedir ingreso de metabolitos no disponibles,
            # permitiendo su secreción.
            lb, ub = 0, COTA

        restricciones[rxn_id] = (lb, ub)

    return restricciones


# ============================================================
# CONSTRUCCIÓN DEL MODELO BASE
# ============================================================

def _modelo_base(com, medio):
    """
    Construye el modelo de comunidad utilizado por MOMA.

    Variables:
        X[i]: abundancia relativa del organismo i.
        V[j]: flujo comunitario de la reacción j.

    Restricciones:
        - Balance de masa comunitario.
        - Restricciones de flujo escaladas por abundancia.
        - Restricciones del medio.
        - Suma de abundancias igual a uno.

    Returns
    -------
    gm : gurobipy.Model
    X : dict
    V : dict
    """

    gm = gp.Model("Community_MOMA")

    organismos = list(com.organisms)

    # --------------------------------------------------------
    # Variables de abundancia
    # --------------------------------------------------------

    X = {
        org: gm.addVar(
            lb=0,
            ub=1,
            name=f"X_{org}"
        )
        for org in organismos
    }

    # --------------------------------------------------------
    # Variables de flujo comunitario
    # --------------------------------------------------------

    V = {
        rxn_id: gm.addVar(
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name=f"V_{rxn_id}"
        )
        for rxn_id in com.merged_model.reactions
    }

    gm.update()

    # --------------------------------------------------------
    # Suma de abundancias
    # --------------------------------------------------------

    gm.addConstr(
        gp.quicksum(X[org] for org in organismos) == 1,
        name="suma_abundancias"
    )

    # --------------------------------------------------------
    # Restricciones del medio comunitario
    # --------------------------------------------------------

    restricciones_medio = constraints_medio(com, medio)

    for rxn_com, (lb, ub) in restricciones_medio.items():

        gm.addConstr(
            V[rxn_com] >= lb,
            name=f"medio_lb_{rxn_com}"
        )

        gm.addConstr(
            V[rxn_com] <= ub,
            name=f"medio_ub_{rxn_com}"
        )

    # --------------------------------------------------------
    # Cotas de reacciones individuales
    # --------------------------------------------------------

    for org, modelo in com.organisms.items():

        for rxn_id, rxn in modelo.reactions.items():

            # La reacción de biomasa se maneja por separado.
            if rxn_id == modelo.biomass_reaction:
                continue

            rxn_com = com.reaction_map.get((org, rxn_id))

            # Omitir reacciones que no estén en el mapa.
            if rxn_com is None:
                continue

            gm.addConstr(
                V[rxn_com] >= rxn.lb * X[org],
                name=f"lb_{org}_{rxn_id}"
            )

            gm.addConstr(
                V[rxn_com] <= rxn.ub * X[org],
                name=f"ub_{org}_{rxn_id}"
            )

    # --------------------------------------------------
    # Balances de masa del modelo comunitario
    # --------------------------------------------------

    for met_id in com.merged_model.metabolites:

        terminos = []

        for rxn_id, rxn in com.merged_model.reactions.items():

            coef = rxn.stoichiometry.get(met_id, 0)

            if coef != 0:
                terminos.append(coef * V[rxn_id])

        if terminos:
            gm.addConstr(
                gp.quicksum(terminos) == 0,
                name=f"balance_{met_id}"
            )

    gm.update()

    return gm, X, V


# ============================================================
# RESTRICCIONES DE CRECIMIENTO
# ============================================================

def _fijar_crecimiento(gm, V, X, com, mu):

    for org, modelo in com.organisms.items():

        rxn_biomasa = modelo.biomass_reaction
        rxn_com = com.reaction_map[(org, rxn_biomasa)]

        gm.addConstr(
            V[rxn_com] == mu * X[org],
            name=f"crecimiento_{org}"
        )


# ============================================================
# OPTIMIZACIÓN
# ============================================================

def _optimizar(gm, tiempo_limite=600, mostrar_log=False, mip_gap=None):
    gm.Params.NonConvex = 2
    gm.Params.OptimalityTarget = 1
    gm.Params.TimeLimit = tiempo_limite
    gm.Params.OutputFlag = int(mostrar_log)

    gm.optimize()

    print(f"Status: {gm.Status}")
    print(f"Soluciones factibles: {gm.SolCount}")

    if gm.SolCount > 0:
        print(f"Valor objetivo: {gm.ObjVal}")


# ============================================================
# COMPROBACIÓN DEL MODELO BASE
# ============================================================

def comprobar_base(
    referencias,
    medio,
    abs_tol=1e-5
):
    """
    Comprueba la consistencia del modelo comunitario
    utilizando la referencia de SteadyCom.

    Verifica factibilidad por debajo del crecimiento máximo
    y falta de factibilidad por encima de este.
    """

    com = referencias["steadycom"]["comunidad"]
    mu_max = referencias["steadycom"]["mu_max"]

    resultados = {}

    for nombre, mu in [
        ("inferior", mu_max - abs_tol),
        ("superior", mu_max + abs_tol)
    ]:

        gm, X, V = _modelo_base(com, medio)

        _fijar_crecimiento(gm, V, X, com, mu)

        status = _optimizar(
            gm,
            mostrar_log=False
        )

        resultados[nombre] = {
            "mu": mu,
            "factible": status in (GRB.OPTIMAL, GRB.SUBOPTIMAL),
            "status": status
        }

        gm.dispose()

    return resultados


# ============================================================
# REFERENCIAS DE FLUJO INDIVIDUALES
# ============================================================

def preparar_referencias_moma(referencias, com):
    """
    Prepara las referencias pFBA individuales de cada organismo.

    Convierte los identificadores de reacción individuales
    a los identificadores correspondientes en la comunidad.

    Returns
    -------
    dict
        refs[organismo][reaccion_comunitaria] = flujo pFBA
    """

    refs = {}

    for org in com.organisms:

        if org not in referencias["pfba"]:
            raise ValueError(
                f"No existe una referencia pFBA para '{org}'."
            )

        flujos = referencias["pfba"][org]["flujos"]

        refs[org] = {}

        for rxn_id, flujo in flujos.items():

            clave = (org, rxn_id)

            if clave not in com.reaction_map:
                continue

            rxn_com = com.reaction_map[clave]

            refs[org][rxn_com] = float(flujo)

    return refs


# ============================================================
# REFERENCIA DE ABUNDANCIAS DE LOS RESIDENTES
# ============================================================

def referencia_abundancias_residentes(
    abundancias,
    residentes
):
    """
    Normaliza las abundancias de SteadyCom considerando
    únicamente los organismos residentes.

    Si no existen residentes, retorna un diccionario vacío.
    """

    if not residentes:
        return {}

    total = sum(
        abundancias.get(org, 0)
        for org in residentes
    )

    if total <= 0:
        raise ValueError(
            "La abundancia total de los residentes debe ser positiva."
        )

    return {
        org: abundancias.get(org, 0) / total
        for org in residentes
    }



# ============================================================
# CLASE DEL PROBLEMA MOMA
# ============================================================
class ProblemaInercia:

    def __init__(
        self,
        com,
        medio,
        refs,
        residentes,
        invasores,
        X_ref=None,
        eps=1e-4,
        pesos="norma",
        todos_presentes=True
    ):
        self.com = com
        self.medio = medio
        self.refs = refs
        self.residentes = list(residentes)
        self.invasores = list(invasores)
        self.eps = eps
        self.pesos_tipo = pesos

        # Validar partición de organismos
        organismos = set(com.organisms.keys())
        particion = set(self.residentes) | set(self.invasores)

        if particion != organismos:
            raise ValueError(
                "Residentes e invasores deben particionar "
                "exactamente los organismos de la comunidad."
            )

        if set(self.residentes) & set(self.invasores):
            raise ValueError("Un organismo no puede ser residente e invasor.")

        if not self.invasores:
            raise ValueError("Debe existir al menos un invasor.")

        if pesos not in PESOS_VALIDOS:
            raise ValueError(
                f"Tipo de pesos inválido. Opciones: {PESOS_VALIDOS}"
            )

        # Abundancias de referencia de los residentes
        if X_ref is None:
            raise ValueError(
                "Debe proporcionarse X_ref con las abundancias "
                "de referencia de los residentes."
            )

        suma_ref = sum(X_ref[org] for org in self.residentes)

        self.X_ref = {
            org: X_ref[org] / suma_ref
            for org in self.residentes
        }

        # Organismos que deben permanecer presentes
        self.presentes = (
            list(com.organisms.keys())
            if todos_presentes
            else self.invasores
        )

        # Construcción del modelo base
        self.gm, self.X, self.V = _modelo_base(com, medio)

        # Variables de flujos específicos para cada organismo
        self.v = {}
        self.reacciones = {}

        for org in com.organisms:
            self.reacciones[org] = list(refs[org].keys())

            for rxn_com in self.reacciones[org]:
                self.v[(org, rxn_com)] = self.gm.addVar(
                    lb=-GRB.INFINITY,
                    ub=GRB.INFINITY,
                    name=f"v_{org}_{rxn_com}"
                )

                # Acoplamiento entre flujo comunitario y específico
                self.gm.addConstr(
                    self.V[rxn_com]
                    == self.X[org] * self.v[(org, rxn_com)],
                    name=f"acoplamiento_{org}_{rxn_com}"
                )

        # Restricciones de presencia
        for org in self.presentes:
            self.gm.addConstr(
                self.X[org] >= self.eps,
                name=f"presencia_{org}"
            )

        # Pesos de cada organismo
        self.w = {}

        for org in com.organisms:
            if pesos == "norma":
                norma = sum(
                    valor ** 2
                    for valor in refs[org].values()
                ) ** 0.5

                self.w[org] = 1.0 / max(norma, 1e-12)

            else:
                self.w[org] = 1.0

        # Función objetivo
        self.costo_residentes = gp.QuadExpr()
        self.costo_invasores = gp.QuadExpr()

        for org in self.residentes:
            for rxn_com in self.reacciones[org]:
                diferencia = (
                    self.V[rxn_com]
                    - self.X_ref[org] * refs[org][rxn_com]
                )

                self.costo_residentes += (
                    self.w[org] * diferencia * diferencia
                )

        for org in self.invasores:
            for rxn_com in self.reacciones[org]:
                diferencia = (
                    self.v[(org, rxn_com)]
                    - refs[org][rxn_com]
                )

                self.costo_invasores += (
                    self.w[org] * diferencia * diferencia
                )

        self.costo_total = (
            self.costo_residentes + self.costo_invasores
        )

        self.gm.setObjective(
            self.costo_total,
            GRB.MINIMIZE
        )

        # Restricciones de crecimiento, que se actualizarán en resolver()
        self.restricciones_crecimiento = {}

        self.gm.update()

    def resolver(
        self,
        mu,
        tiempo_limite=600,
        mostrar_log=False
    ):
        """
        Resuelve el problema MOMA para una tasa de crecimiento mu.

        Devuelve la mejor solución factible encontrada, aunque Gurobi
        no haya podido certificar optimalidad.
        """

        # Eliminar restricciones de crecimiento anteriores
        if self.restricciones_crecimiento:
            self.gm.remove(
                list(self.restricciones_crecimiento.values())
            )

        self.restricciones_crecimiento = {}

        # Imponer crecimiento de cada organismo
        for org, modelo in self.com.organisms.items():

            rxn_biomasa = modelo.biomass_reaction
            rxn_com = self.com.reaction_map[(org, rxn_biomasa)]

            restriccion = self.gm.addConstr(
                self.V[rxn_com] == mu * self.X[org],
                name=f"crecimiento_{org}"
            )

            self.restricciones_crecimiento[org] = restriccion

        self.gm.update()

        # Configuración del solver
        self.gm.Params.NonConvex = 2
        self.gm.Params.OptimalityTarget = 1
        self.gm.Params.TimeLimit = tiempo_limite
        self.gm.Params.OutputFlag = int(mostrar_log)

        # Resolver
        self.gm.optimize()

        status = self.gm.Status
        sol_count = self.gm.SolCount

        resultado = {
            "mu": mu,
            "status": status,
            "factible": sol_count > 0,
            "objetivo": None,
            "cost_residentes": None,
            "cost_invasores": None,
            "cost_total": None,
            "abundancias": None,
            "flujos": None,
            "flujos_especificos": None,
            "soluciones": sol_count,
            "tiempo": self.gm.Runtime
        }

        # Si no existe solución factible, devolver diagnóstico
        if sol_count == 0:
            print(
                f"Sin solución factible | "
                f"mu={mu:.6f} | status={status}"
            )
            return resultado

        # Recuperar solución
        resultado["objetivo"] = self.gm.ObjVal

        resultado["abundancias"] = {
            org: self.X[org].X
            for org in self.com.organisms
        }

        resultado["flujos"] = {
            rxn: var.X
            for rxn, var in self.V.items()
        }

        resultado["flujos_especificos"] = {
            (org, rxn): var.X
            for (org, rxn), var in self.v.items()
        }

        resultado["cost_residentes"] = self.costo_residentes.getValue()
        resultado["cost_invasores"] = self.costo_invasores.getValue()
        resultado["cost_total"] = self.costo_total.getValue()

        print(
            f"mu={mu:.6f} | "
            f"status={status} | "
            f"soluciones={sol_count} | "
            f"objetivo={resultado['objetivo']:.6f}"
        )

        return resultado

    def cerrar(self):
        """Libera los recursos del modelo de Gurobi."""
        if self.gm is not None:
            self.gm.dispose()
            self.gm = None


# ============================================================
# BARRIDO MOMA
# ============================================================

def barrido_moma(
    referencias,
    medio,
    alphas,
    residentes,
    invasores,
    X_ref=None,
    eps=1e-4,
    pesos="norma",
    coexistencia=True,
    abs_tol=1e-5,
    max_iters=60,
    tiempo_limite=600,
    mostrar_log=False
):
    """
    Ejecuta un barrido de crecimiento comunitario mediante MOMA.

    Parameters
    ----------
    referencias : dict
        Referencias SteadyCom y pFBA.
    medio : dict
        Condiciones del medio.
    alphas : iterable
        Fracciones del crecimiento máximo.
    residentes : list
        Organismos residentes. Puede ser una lista vacía.
    invasores : list
        Organismos invasores. Debe contener al menos uno.
    X_ref : dict, optional
        Abundancias relativas de referencia de los residentes.
    eps : float
        Abundancia mínima para organismos presentes.
    pesos : str
        "norma" o "uno".
    coexistencia : bool
        Si es True, calcula el techo exigiendo presencia de todos.
    """

    steadycom_ref = referencias["steadycom"]

    com = steadycom_ref["comunidad"]
    mu_steadycom = steadycom_ref["mu_max"]

    abundancias_steadycom = steadycom_ref.get(
        "abundancias",
        {}
    )

    residentes = list(residentes)
    invasores = list(invasores)

    # --------------------------------------------------------
    # Validar grupos
    # --------------------------------------------------------

    if not invasores:
        raise ValueError(
            "Debe especificarse al menos un organismo invasor."
        )

    if set(residentes) & set(invasores):
        raise ValueError(
            "Los grupos de residentes e invasores deben ser disjuntos."
        )

    if set(residentes) | set(invasores) != set(com.organisms):
        raise ValueError(
            "La unión de residentes e invasores debe contener "
            "exactamente todos los organismos de la comunidad."
        )

    # --------------------------------------------------------
    # Preparar referencias pFBA
    # --------------------------------------------------------

    refs_moma = preparar_referencias_moma(
        referencias,
        com
    )

    
    # --------------------------------------------------------
    # Referencia de abundancias previa a la invasión
    # --------------------------------------------------------

    referencia_residentes = None

    if not residentes:
        X_ref = {}

    elif X_ref is None:

        # Seleccionar exclusivamente los modelos residentes.
        modelos_residentes = {
            org: referencias["modelos"][org]
            for org in residentes
        }

        # Ejecutar SteadyCom sin incluir a los invasores.
        referencia_residentes = calcular_referencia_comunitaria(
            modelos=modelos_residentes,
            medio=medio,
            abs_tol=abs_tol,
            max_iters=max_iters
        )

        # Estas son las abundancias previas a la invasión.
        X_ref = referencia_residentes["abundancias"].copy()

    else:
        # Permitir referencias manuales.
        if set(X_ref) != set(residentes):
            raise ValueError(
                "X_ref debe contener exactamente los residentes."
            )

        X_ref = {
            org: float(X_ref[org])
            for org in residentes
        }

    # Verificación y normalización por posibles errores numéricos.
    if residentes:
        suma = sum(X_ref.values())

        if suma <= 0:
            raise ValueError(
                "La suma de abundancias residentes debe ser positiva."
            )

        X_ref = {
            org: valor / suma
            for org, valor in X_ref.items()
        }

    # --------------------------------------------------------
    # Preparar abundancias de referencia
    # --------------------------------------------------------

    if X_ref is None:

        X_ref = referencia_abundancias_residentes(
            abundancias_steadycom,
            residentes
        )

    elif not residentes:

        X_ref = {}


    mu_techo = referencias["steadycom"]["mu_max"]

    problema = ProblemaInercia(
        com=com,
        medio=medio,
        refs=refs_moma,
        X_ref=X_ref,
        residentes=residentes,
        invasores=invasores,
        pesos=pesos,
        eps=eps,
        todos_presentes=coexistencia
    )


    # Barrido
    filas = []

    try:
        for alpha in alphas:

            mu = float(alpha) * mu_techo

            resultado = problema.resolver(
                mu=mu,
                tiempo_limite=tiempo_limite,
                mostrar_log=mostrar_log
            )

            print(
                f"alpha={alpha:.2f} | "
                f"mu={mu:.6f} | "
                f"status={resultado['status']} | "
            )

            fila = {
                "alpha": float(alpha),
                "mu": mu,
                "status": resultado["status"],
                "factible": resultado["factible"],
                "objetivo": resultado["objetivo"],
                "cost_residentes": resultado["cost_residentes"],
                "cost_invasores": resultado["cost_invasores"],
                "cost_total": resultado["cost_total"],
                "abundancias": resultado["abundancias"],
                "flujos": resultado["flujos"]
            }

            filas.append(fila)

    finally:
        pesos_finales = problema.w.copy()
        problema.cerrar()

    # --------------------------------------------------------
    # Resultado final
    # --------------------------------------------------------

    return {
        "mu_steadycom": mu_steadycom,
        "mu_techo": mu_techo,
        "abundancias_steadycom": abundancias_steadycom,
        "abundancias_referencia_residentes": X_ref,
        "residentes": residentes,
        "invasores": invasores,
        "filas": filas,
        "comunidad": com,
        "referencias": referencias,
        "refs_moma": refs_moma,
        "pesos": pesos_finales,
        "steadycom_residentes": referencia_residentes
    }