
"""
Cálculo de referencias metabólicas para organismos individuales mediante pFBA y de referencias 
comunitarias mediante SteadyCom.

- pFBA individual:
    Calcula el crecimiento máximo y los flujos metabólicos de cada
    organismo por separado.

- SteadyCom comunitario:
    Calcula la tasa de crecimiento máxima de la comunidad y las
    abundancias relativas de sus organismos.

"""

import cobra
from cobra.flux_analysis import pfba
from .steadycom_adapted import SteadyCom
from reframed import from_cobrapy
from reframed.community.model import Community
from reframed.core.model import ReactionType
import contextlib
import io

# Parámetros globales
COTA = 100    # cota de intercambios no listados / secrecion y flujos internos no acotados


def aplicar_medio(modelo, medio):
    """
    Aplica las restricciones de un medio de cultivo a un modelo COBRApy.

    ENTRADAS
    ----------
    modelo : cobra.Model
        Modelo metabólico individual.
    medio : dict
        Diccionario con formato:
        {
            "glc__D_e": (-10, 1000),
            "o2_e": (-15, 1000),
            ...
        }

    SALIDAS
    -------
    modelo : cobra.Model
        Copia del modelo con las restricciones ambientales aplicadas.
    """

    modelo = modelo.copy()
    for reaccion in modelo.exchanges:
        # para robustez
        if len(reaccion.metabolites) != 1:
            raise ValueError(f"El intercambio {reaccion.id} no contiene exactamente un metabolito.")

        metabolito = next(iter(reaccion.metabolites)).id
        # Si el metabolito no está en el medio, se impide su captación.
        lb, ub = medio.get(metabolito, (0.0, reaccion.upper_bound))
        reaccion.bounds = (float(lb), float(ub))

    return modelo


def calcular_referencia_pfba(modelo, medio):
    """
    Calcula la referencia metabólica de un organismo mediante pFBA.

    ENTRADAS
    ----------
    modelo : cobra.Model
        Modelo metabólico individual.
    medio : dict
        Medio de cultivo del organismo.

    SALIDAS
    -------
    dict
        mu_max : crecimiento máximo.
        flujos : flujos metabólicos de la solución pFBA.
        solucion : solución completa de COBRApy.
        modelo : modelo con las restricciones del medio.
    """

    modelo_medio = aplicar_medio(modelo, medio)

    # Crecimiento máximo mediante FBA.
    solucion_fba = modelo_medio.optimize()

    if solucion_fba.status != "optimal":
        # para robustez
        raise RuntimeError(
            f"FBA individual infactible para {modelo.id}. "
            f"Estado: {solucion_fba.status}"
        )

    mu_max = float(solucion_fba.objective_value)

    # pFBA: minimiza la suma de los flujos absolutos manteniendo el óptimo de crecimiento.
    solucion_pfba = pfba(modelo_medio, fraction_of_optimum=1.0)
    if solucion_pfba.status != "optimal":
        # para robustez
        raise RuntimeError(
            f"pFBA infactible para {modelo.id}. "
            f"Estado: {solucion_pfba.status}"
        )

    fluxes = solucion_pfba.fluxes.to_dict()

    return {
        "mu_max": mu_max,
        "flujos": fluxes,
        "solucion": solucion_pfba,
        "modelo": modelo_medio
    }


def calcular_referencias_individuales(modelos, medios_ref):
    """
    Calcula las referencias pFBA de todos los organismos.

    ENTRADAS
    ----------
    modelos : dict
        Diccionario {organismo: modelo COBRApy}.
    medios_ref : dict
        Diccionario {organismo: medio individual}.

    Si un organismo no tiene un medio específico, se utiliza
    un medio vacío, que impide la captación de metabolitos.

    SALIDAS
    -------
    dict
        Referencias individuales (PFBA) de cada organismo.
    """

    referencias = {}
    for organismo, modelo in modelos.items():
        if organismo not in medios_ref:
            # para robustez
            raise ValueError(
                f"No se especificó un medio de referencia para {organismo}."
            )

        medio = medios_ref[organismo]
        referencias[organismo] = calcular_referencia_pfba(modelo, medio)

    return referencias


def calcular_referencia_comunitaria(modelos, medio, abs_tol=1e-5, max_iters=60):
    """
    Calcula la referencia de una comunidad mediante SteadyCom.

    Utiliza:
        - SteadyCom() de steadycom_adapted.py

    ENTRADAS
    ----------
    modelos : dict
        Diccionario {organismo: modelo COBRApy}.
    medio : dict
        Medio de cultivo compartido por la comunidad.
    abs_tol : float
        Tolerancia absoluta para la búsqueda de crecimiento.
    max_iters : int
        Número máximo de iteraciones de SteadyCom.

    SALIDAS
    -------
    dict
        mu_max : crecimiento máximo comunitario.
        abundancias : abundancias relativas de los organismos.
        solucion : solución completa de SteadyCom.
        comunidad : objeto Community de ReFramed.
    """

    # Construcción de la comunidad.
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

    comunidad = Community("comunidad", [a_reframed(m, org) for org, m in modelos.items()])
    if not comunidad.merged_model.get_exchange_reactions():
        raise RuntimeError("El modelo fusionado no tiene intercambios comunitarios (R_EX_*): "
                        "revisar compartimento externo y reaction_type.")

    # Aplicación de las restricciones ambientales.
    merged = comunidad.merged_model
    restricciones, usados = {}, set()
    for rid in merged.get_exchange_reactions():
        met = next(iter(merged.reactions[rid].stoichiometry))
        base = met[2:] if met.startswith("M_") else met
        lb, ub = medio.get(base, (0.0, COTA))
        restricciones[rid] = (max(float(lb), -COTA), min(float(ub), COTA))
        usados.add(base)
    faltan = set(medio) - usados
    if faltan:
        raise ValueError(f"Metabolitos del medio ausentes en la comunidad: {sorted(faltan)}")

    # Resolución mediante la implementación adaptada
    # Silenciar los mensajes de SteadyCom y del solver
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):

        solucion = SteadyCom(
            comunidad,
            constraints=restricciones,
            abs_tol=abs_tol,
            max_iters=max_iters,
            verbose=False
        )    
    mu_max = float(solucion.growth)
    abundancias = {organismo: float(solucion.abundance[organismo]) for organismo in modelos}

    return {
        "mu_max": mu_max,
        "abundancias": abundancias,
        "solucion": solucion,
        "comunidad": comunidad
    }


def calcular_referencias(modelos, medio_comunidad, medios_individuales, abs_tol=1e-5, max_iters=60):
    """
    Calcula las referencias individuales pFBA y comunitaria SteadyCom.

    ENTRADAS
    ----------
    modelos : dict
        Modelos metabólicos de los organismos.
    medio_comunidad : dict
        Medio compartido de la comunidad.
    medios_individuales : dict
        Medios de referencia individuales de cada organismo.
    abs_tol : float
        Tolerancia de SteadyCom.
    max_iters : int
        Iteraciones máximas de SteadyCom.

    SALIDAS
    -------
    dict
        {
            "pfba": referencias individuales,
            "steadycom": referencia comunitaria
        }
    """
    referencias_pfba = calcular_referencias_individuales(modelos, medios_individuales)
    referencia_community = calcular_referencia_comunitaria(modelos, medio_comunidad, abs_tol=abs_tol, max_iters=max_iters)
    
    return {
        "pfba": referencias_pfba,
        "steadycom": referencia_community,
        "modelos": modelos
    }



def imprimir_referencias(referencias):
    """
    Imprime un resumen de los crecimientos y abundancias calculados, 
    a partir de la salida de calcular_referencias().
    """

    print("\n========== REFERENCIAS pFBA INDIVIDUALES ==========")
    print(f"{'Organismo':<15}{'Crecimiento máximo':>22}")
    for organismo, ref in referencias["pfba"].items():
        print(
            f"{organismo:<15}"
            f"{ref['mu_max']:>22.6f}"
        )

    print("\n========== REFERENCIA STEADYCOM COMUNITARIA ==========")
    ref_com = referencias["steadycom"]
    print(f"Crecimiento máximo comunitario: {ref_com['mu_max']:.6f}")
    print(f"{'Organismo':<15}{'Abundancia relativa':>22}")
    for organismo, abundancia in ref_com["abundancias"].items():
        print(
            f"{organismo:<15}"
            f"{abundancia:>22.6f}"
        )