import contextlib, math, os, sys

@contextlib.contextmanager
def silencio():
    sys.stdout.flush()
    nulo = os.open(os.devnull, os.O_WRONLY); g1 = os.dup(1)
    os.dup2(nulo, 1)
    try:
        yield
    finally:
        sys.stdout.flush(); os.dup2(g1, 1); os.close(nulo); os.close(g1)

def tabla(res, nd=3):
    print(f"mu_techo = {res['mu_techo']:.4f} 1/h")
    print(f"{'alpha':>6} | {'orden 1: (B1+B2) → B3':<34} | {'orden 2: B3 → (B1+B2)':<34} | {'PE':>5}")
    def celda(r):
        if r["estado"] == "ok":
            return " ".join(f"{k}={v:.{nd}f}" for k, v in r["X"].items())
        return r["estado"] + (f" (gurobi {r['gstat']})" if r["estado"] == "error" else "")
    for f in res["filas"]:
        pe = "-" if f["PE"] is None else f"{f['PE']:.3f}"
        print(f"{f['alpha']:>6.3f} | {celda(f['orden1']):<34} | {celda(f['orden2']):<34} | {pe:>5}")

def tabla_delta(res, nd=3):
    """Δ (ec. 15) y costo metabólico (ec. 11). Δ del orden 2 es 0 por construcción (1 residente)."""
    fmt = lambda x: "-" if x is None else f"{x:.{nd}f}"
    sci = lambda x: "-" if x is None else f"{x:.2e}"
    print(f"{'alpha':>6} | {'Δ orden 1':>10} | {'costo o1':>10} | {'costo o2':>10}")
    for f in res["filas"]:
        print(f"{f['alpha']:>6.3f} | {fmt(f['d1']):>10} | "
              f"{sci(f['orden1']['costo']):>10} | {sci(f['orden2']['costo']):>10}")