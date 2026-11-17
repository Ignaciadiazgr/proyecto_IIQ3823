"""
Este archivo contiene la implementación de SteadyCom adaptada para el proyecto, 
basada en la versión de SteadyCom de reframed (https://github.com/cdanielmachado/reframed.git).

Se han realizado modificaciones para mejorar la compatibilidad con el proyecto y
para facilitar la integración con otras funciones y módulos del proyecto.
Se han modificado las funciones auxiliares para la construcción del problema de optimización,
la búsqueda de la tasa de crecimiento y la verificación de la factibilidad de las soluciones.
Se han agregado comentarios para mejorar la comprensión del código.

Referencia:
Chan, S. H. J., Simons, M. N., & Maranas, C. D. (2017). SteadyCom: Predicting microbial
abundances while ensuring community stability. PLOS Computational Biology, 13 (5),
1-25. https://doi.org/10.1371/journal.pcbi.1005539
"""

from reframed.community.solution import CommunitySolution
from reframed.solvers.solution import Status
from reframed.core.model import ReactionType
from reframed.solvers import solver_instance
from warnings import warn
from math import inf, isinf

def es_intercambio(rid, reaction):
    return (reaction.reaction_type == ReactionType.EXCHANGE or (rid.startswith("EX_") and len(reaction.stoichiometry) == 1))
    

def SteadyCom(community, constraints=None, solver=None, *, abs_tol=1e-3, max_iters=20, verbose=False):
    """ Implementación de SteadyCom (Chan et al 2017).

    Entradas:
        community (CommunityModel): modelo de comunidad
        constraints (dict): restricciones ambientales o adicionales (opcional)
        solver (Solver): instancia del solver inicializada con el modelo, para mejorar la velocidad (opcional)
        abs_tol (float): tolerancia absoluta para la búsqueda binaria (opcional)
        max_iters (int): número máximo de iteraciones para la búsqueda binaria (opcional)
        verbose (bool): si se desea imprimir información detallada durante la ejecución (opcional)

    Salidas:
        CommunitySolution: solución de la comunidad con abundancias y tasas de crecimiento
    """

    if solver is None:
        solver = build_problem(community)

    objective = {community.merged_model.biomass_reaction: 1}

    sol = binary_search(solver, objective, minimize=False, constraints=constraints,
                        abs_tol=abs_tol, max_iters=max_iters, verbose=verbose)

    growth_id = community.merged_model.biomass_reaction
    mu = sol.values.get(growth_id, None)

    if mu is None or mu <= 1e-10:
        raise RuntimeError(
            "SteadyCom no encontró crecimiento positivo. "
            "Revisar medio, restricciones y factibilidad."
        )

    solution = CommunitySolution(community, sol.values)
    solution.solver = solver

    return solution


def build_problem(community, growth=1, bigM=1000):

    solver = solver_instance()
    model = community.merged_model

    # create biomass variables
    for org_id in community.organisms:
        solver.add_variable(f"x_{org_id}", 0, 1)

    # create all community reactions
    for r_id, reaction in model.reactions.items():
        if es_intercambio(r_id, reaction):
            solver.add_variable(r_id, reaction.lb, reaction.ub)
        else:
            lb = -inf if reaction.lb < 0 else 0
            ub = inf if reaction.ub > 0 else 0
            solver.add_variable(r_id, lb, ub)

    solver.update()

    # sum biomass = 1
    solver.add_constraint("abundance", {f"x_{org_id}": 1 for org_id in community.organisms},
                          rhs=1)

    # S.v = 0
    table = model.metabolite_reaction_lookup()
    for m_id in model.metabolites:
        solver.add_constraint(m_id, table[m_id])

    # organism-specific constraints
    for org_id, organism in community.organisms.items():

        for r_id, reaction in organism.reactions.items():
            if (org_id, r_id) not in community.reaction_map:
                continue

            new_id = community.reaction_map[(org_id, r_id)]

            # growth = mu * X
            if r_id == organism.biomass_reaction:
                solver.add_constraint(f"g_{org_id}", {f"x_{org_id}": growth, new_id: -1})
            # lb * X < R < ub * X
            else:
                lb = -bigM if isinf(reaction.lb) else reaction.lb
                ub = bigM if isinf(reaction.ub) else reaction.ub

                if lb != 0:
                    solver.add_constraint(f"lb_{new_id}", {f"x_{org_id}": lb, new_id: -1}, '<', 0)

                if ub != 0:
                    solver.add_constraint(f"ub_{new_id}", {f"x_{org_id}": ub, new_id: -1}, '>', 0)

    solver.update()

    def update_growth(value):
        model = solver.problem

        model.update()

        for x in community.organisms:
            constraint = model.getConstrByName(f"g_{x}")
            variable = model.getVarByName(f"x_{x}")

            if constraint is None:
                raise ValueError(f"No se encontró la restricción g_{x}")

            if variable is None:
                raise ValueError(f"No se encontró la variable x_{x}")

            model.chgCoeff(constraint, variable, value)

        model.update()

    solver.update_growth = update_growth

    return solver


def binary_search(solver, objective, obj_frac=1, minimize=False, max_iters=20, abs_tol=1e-3, constraints=None, verbose=False):
    previous_value = 0
    value = 1
    fold = 2
    feasible = False
    last_feasible = 0

    for i in range(max_iters):
        diff = value - previous_value

        if diff < abs_tol:
            break

        if feasible:
            last_feasible = value
            previous_value = value
            value = fold*diff + value
        else:
            if i > 0:
                fold = 0.5
            value = fold*diff + previous_value

        solver.update_growth(value)

        sol = solver.solve(
            objective,
            get_values=False,
            minimize=minimize,
            constraints=constraints
        )

        print(
            f"mu probado = {value}, "
            f"status = {sol.status}, "
            f"last_feasible = {last_feasible}"
        )

        feasible = sol.status == Status.OPTIMAL

    if feasible:
        solver.update_growth(obj_frac * value)
    else:
        solver.update_growth(obj_frac * last_feasible)

    sol = solver.solve(objective, minimize=minimize, constraints=constraints)

    if i == max_iters - 1:
        warn("Max iterations exceeded.")

    return sol