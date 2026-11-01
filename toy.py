import community_moma as cm
import cobra
from reframed import from_cobrapy, FBA
import matplotlib.pyplot as plt
from utils import tabla, silencio
import figuras

# FBA de cobrapy para verificar resultados obtenidos con reframed
def fba_cobra(modelo, medio):
    m = modelo.copy()
    for r in m.exchanges:
        met = next(iter(r.metabolites)).id
        lb, ub = medio.get(met, (0.0, r.upper_bound))
        r.bounds = (lb, ub)
    return m.slim_optimize()

# FBA de reframed para verificar adaptación de SteadyCom
def fba_reframed(modelo, medio):
    m = from_cobrapy(modelo.copy())
    aplicados = []

    # Recorrer los intercambios de metabolitos identificados por COBRApy
    for rxn_cobra in modelo.exchanges:
        rid = rxn_cobra.id
        if rid not in m.reactions:
            raise ValueError(
                f"La reacción {rid} no existe en el modelo ReFramed"
            )
        r = m.reactions[rid]

        # Obtener el metabolito asociado al intercambio
        if len(r.stoichiometry) != 1:
            raise ValueError(
                f"Se esperaba un metabolito en {rid}: {r.stoichiometry}"
            )
        met_id = next(iter(r.stoichiometry))

        if met_id in medio:
            lb, ub = medio[met_id]
            r.lb = float(lb)
            r.ub = float(ub)
            aplicados.append(rid)
        else:
            # Impedir la captación de metabolitos no incluidos en el medio
            r.lb = 0.0

    # Resolver FBA
    sol = FBA(m)
    return sol, aplicados

# Modelo de comunidad de juguete de 3 especies diferentes, 
# en base a E. coli K-12 (modelo "textbook" de COBRApy)
def modelo_toy():
    base = cobra.io.load_model("textbook")
    B1, B2, B3 = base.copy(), base.copy(), base.copy()
    # B1: auxótrofo de glutamato
    B1.reactions.get_by_id("GLUDy").bounds = (0, 0)
    B1.reactions.get_by_id("GLUSy").bounds = (0, 0)
    # B2: no usa glucosa, vive del acetato de B1
    B2.reactions.get_by_id("GLCpts").bounds = (0, 0)
    # B3: sin respiración, compite con B1 por glucosa
    B3.reactions.get_by_id("CYTBD").bounds = (0, 0)
    return {"B1": B1, "B2": B2, "B3": B3}

# Medio de cultivo simple
MEDIO = {
    "glc__D_e": (-10, 1000),
    "o2_e": (-15, 1000),
    "nh4_e": (-1000, 1000),
    "pi_e": (-1000, 1000),
    "h2o_e": (-1000, 1000),
    "h_e": (-1000, 1000),
    "co2_e": (-1000, 1000)
}

# Medios de referencia para cada especie
MEDIOS_REF = {
    "B1": {**MEDIO, "glu__L_e": (-5, 1000)},
    "B2": {**MEDIO, "ac_e": (-10, 1000)}
}

# Programa principal
if __name__ == "__main__":
    modelos = modelo_toy()
    alphas = [1.0, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4]

    with cm.silencio():
        # Validación de un organismo: cobrapy | FBA del modelo fusionado | SteadyCom adaptado
        val = []
        for k, m in modelos.items():
            med = MEDIOS_REF.get(k, MEDIO)
            com = cm.crear_comunidad({k: m})
            cons = cm.constraints_medio(com, med)
            ref = cm.referencia({k: m}, med)
            ok_mu, ok_sobre = cm.comprobar_base(com, med, ref["mu"], 1e-5)
            val.append((k, fba_cobra(m, med), FBA(com.merged_model, constraints=cons).fobj,
                        ref["mu"], ok_mu, ok_sobre))

        # Barridos: pesos de la ec. 14 (informe) y pesos unitarios (sensibilidad)
        res_n = cm.barrido_prioridad(modelos, MEDIO, alphas, medios_ref=MEDIOS_REF, pesos="norma")
        res_u = cm.barrido_prioridad(modelos, MEDIO, alphas, medios_ref=MEDIOS_REF, pesos="uno")

    print(f"{'':3}{'cobrapy':>9}{'FBA fusion':>12}{'SteadyCom':>11}  base(mu, mu+)")
    for k, a, b, c, d, e in val:
        print(f"{k:3}{a:9.4f}{b:12.4f}{c:11.4f}  ({d}, {e})")

    print("\n=== Pesos normalizados w_i = 1/||v*_i||^2 (ec. 14) ===")
    cm.tabla(res_n)
    print("\n=== Pesos unitarios w_i = 1 (sensibilidad) ===")
    cm.tabla(res_u)

    print("\nPesos w_i usados (normalizados):")
    for orden in ("orden1", "orden2"):
        print(f"  {orden}:", {o: f"{w:.2e}" for o, w in res_n["pesos"][orden].items()})

    # Costo por especie en alpha = 0.9 (muestra quién paga la reorganización)
    f = next(f for f in res_n["filas"] if abs(f["alpha"] - 0.9) < 1e-9)
    for orden in ("orden1", "orden2"):
        r = f[orden]
        if r["estado"] == "ok":
            print(f"\nCosto ponderado por especie, alpha=0.9, {orden}:",
                  {o: f"{c:.3e}" for o, c in r["costo_esp"].items()})

    # Figuras
    figuras.figura_resultados(res_n, res_u, "fig_resultados", escala="log")
    figuras.figura_resultados(res_n, res_u, "fig_resultados_lineal", escala="lineal")
    figuras.figura_sensibilidad(res_u, "fig_sensibilidad")