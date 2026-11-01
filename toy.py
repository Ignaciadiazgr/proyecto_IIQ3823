import community_moma as cm
import cobra
import steadycom as sc
from steadycom import SteadyComParams

# FBA para verificar resultados
def fba_cobra(modelo, medio):
    m = modelo.copy()
    for r in m.exchanges:
        met = next(iter(r.metabolites)).id
        lb, ub = medio.get(met, (0.0, r.upper_bound))
        r.bounds = (lb, ub)
    return m.slim_optimize()

# Para modelos toy con cobra
def modelo_toy():
    base = cobra.io.load_model("textbook")
    B1, B2, B3 = base.copy(), base.copy(), base.copy()
    # B1: auxótrofo de glutamato (sin glutamato deshidrogenasa ni glutamato sintasa)
    B1.reactions.get_by_id("GLUDy").bounds = (0, 0)
    B1.reactions.get_by_id("GLUSy").bounds = (0, 0)
    # B2: no usa glucosa, vive del acetato de B1
    B2.reactions.get_by_id("GLCpts").bounds = (0, 0)
    # B3: sin respiración, compite con B1 por glucosa
    B3.reactions.get_by_id("CYTBD").bounds = (0, 0)
    return {"B1": B1, "B2": B2, "B3": B3}

MEDIO = {"glc__D_e": (-10, 1000), "o2_e": (-15, 1000),
         "nh4_e": (-1000, 1000), "pi_e": (-1000, 1000), "h2o_e": (-1000, 1000),
         "h_e": (-1000, 1000), "co2_e": (-1000, 1000)}
# ni acetato ni glutamato están en el medio común: solo existen por alimentación cruzada

# Referencias "en solitario" (cada especie sola, con lo que necesita aportado externamente)
MEDIOS_REF = {
    "B1": {**MEDIO, "glu__L_e": (-5, 1000)},
    "B2": {**MEDIO, "ac_e": (-10, 1000)},
}


# Programa principal
if __name__ == "__main__":
    with cm.silencio():
        B = modelo_toy()
        par = SteadyComParams(gr_tol=1e-6, feas_tol=1e-6)
        val = {}
        for k in B:
            med = MEDIOS_REF.get(k, MEDIO)
            val[k] = (cm.referencia({k: B[k]}, med, par)["mu"], fba_cobra(B[k], med))
        libre = cm.referencia(B, MEDIO, par) if False else None   # ver nota abajo
        com = sc.build_steadycom_model(list(B.values()), list(B.keys()))
        libre = cm.SteadyCom(com, medium=MEDIO, params=par, close_other_exchanges=True)
        res = cm.barrido_prioridad(B, MEDIO, alphas=[0.999, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4],
                                medios_ref=MEDIOS_REF, params=par)

    print("Validación (SteadyCom 1 sp | cobrapy FBA)")
    for k, (a, b) in val.items():
        print(f"  {k}: {a:.6f} | {b:.6f}")
    print("\nSteadyCom libre de los 3:", {k: round(v, 4) for k, v in libre.abundance.items()},
        f"mu={libre.growth_rate:.4f}")
    print()
    print("Efecto prioridad (inercia)")
    cm.tabla(res)