import matplotlib.pyplot as plt

COLORES = {"B1": "#3BA3DF", "B2": "#DB33A6", "B3": "#009E73"}
ETIQ_N = r"$w_i=1/\|v_i^*\|^2$ (ec. 14)"
ETIQ_U = r"$w_i=1$ (sensibilidad)"


def _serie(res, orden):
    filas = [f for f in res["filas"] if f[orden]["estado"] == "ok"]
    if not filas:
        return [], [], {}
    orgs = list(filas[0][orden]["X"])
    return ([f["alpha"] for f in filas], orgs,
            {o: [f[orden]["X"][o] for f in filas] for o in orgs})


def _panel_abundancias(ax, res, orden, titulo, escala, eps):
    a, orgs, ser = _serie(res, orden)
    if not a:
        ax.set_title(titulo + " (sin soluciones)"); return
    if escala == "log":
        for o in orgs:
            ax.plot(a, [max(x, 1e-6) for x in ser[o]], "o-", color=COLORES.get(o), label=o, ms=4)
        ax.set_yscale("log"); ax.set_ylim(5e-5, 1.5)
        ax.axhline(eps, ls=":", c="gray", lw=1)
        ax.text(0.02, eps * 1.15, r"$\epsilon$", transform=ax.get_yaxis_transform(), color="gray")
    else:
        # stackplot necesita x ordenado
        idx = sorted(range(len(a)), key=lambda i: a[i])
        ax.stackplot([a[i] for i in idx], *[[ser[o][i] for i in idx] for o in orgs],
                     labels=orgs, colors=[COLORES.get(o) for o in orgs])
        ax.set_ylim(0, 1)
    ax.set(title=titulo, xlabel=r"$\alpha$", ylabel="Fracción de biomasa")
    ax.invert_xaxis()


def _panel_pe(ax, res_n, res_u):
    for res, etiq, c in ((res_n, ETIQ_N, "#0072B2"), (res_u, ETIQ_U, "#E69F00")):
        if res is None:
            continue
        f = [x for x in res["filas"] if x["PE"] is not None]
        if not f:
            continue
        ax.plot([x["alpha"] for x in f], [x["PE"] for x in f], "o-", color=c, label=etiq, ms=4)
        base = next((x["PE"] for x in f if abs(x["alpha"] - 1.0) < 1e-9), None)
        if base is not None:
            ax.axhline(base, ls="--", color=c, lw=1, alpha=.6)
    ax.set(title="Efecto prioridad", xlabel=r"$\alpha$", ylabel="PE")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)


def _guardar(fig, salida):
    fig.savefig(f"{salida}.png", dpi=300, bbox_inches="tight")


def figura_resultados(res_n, res_u=None, salida="fig_resultados", escala="lineal", eps=1e-4):
    """Pesos normalizados (ec. 14) en los paneles 1-2; PE de ambos pesos en el panel 3."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    _panel_abundancias(ax[0], res_n, "orden1", r"Orden 1: (B1+B2) $\rightarrow$ B3", escala, eps)
    _panel_abundancias(ax[1], res_n, "orden2", r"Orden 2: B3 $\rightarrow$ (B1+B2)", escala, eps)
    ax[1].legend(frameon=False, fontsize=8)
    _panel_pe(ax[2], res_n, res_u)
    _guardar(fig, salida)
    return fig


def figura_sensibilidad(res_u, salida="fig_sensibilidad", escala="log", eps=1e-4):
    """Abundancias con pesos unitarios."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.9), constrained_layout=True)
    _panel_abundancias(ax[0], res_u, "orden1", r"Orden 1 ($w_i=1$)", escala, eps)
    _panel_abundancias(ax[1], res_u, "orden2", r"Orden 2 ($w_i=1$)", escala, eps)
    ax[1].legend(frameon=False, fontsize=8)
    _guardar(fig, salida)
    return fig