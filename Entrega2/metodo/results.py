"""
Funciones para organizar, imprimir y graficar la comparación SteadyCom vs Community MOMA.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def resultados_a_dataframe(resultados):
    """Una fila por alpha, con abundancias MOMA y referencias SteadyCom."""
    abund_sc = resultados["abundancias_steadycom"]
    organismos = list(abund_sc)
    registros = []

    for fila in resultados["filas"]:
        r = fila["resultado"]
        registro = {
            "alpha": fila["alpha"],
            "mu": fila["mu"],
            "status": r["status_texto"],
            "factible": r["solucion_factible"],
            "objetivo": r["objetivo"],
            "tiempo_s": r.get("tiempo"),
        }
        abund_moma = r.get("abundancias") or {}
        for org in organismos:
            registro[f"{org}_MOMA"] = abund_moma.get(org)
            registro[f"{org}_SteadyCom"] = abund_sc[org]
        registros.append(registro)

    return pd.DataFrame(registros)


def imprimir_resumen(resultados, decimales=6):
    """Imprime los valores de referencia y devuelve el DataFrame."""
    print("=" * 65)
    print("COMPARACIÓN STEADYCOM vs COMMUNITY MOMA")
    print("=" * 65)
    print(f"Mu máximo SteadyCom reportado : {resultados['mu_steadycom']:.{decimales}f}")
    print(f"Mu techo con coexistencia     : {resultados['mu_techo']:.{decimales}f}")
    print("\nAbundancias de referencia SteadyCom:")
    for org, x in resultados["abundancias_steadycom"].items():
        print(f"  {org}: {x:.{decimales}f}")

    df = resultados_a_dataframe(resultados)
    factibles = df[df["factible"] == True]
    if not factibles.empty:
        mejor = factibles.loc[factibles["mu"].idxmax()]
        print(f"\nMayor mu MOMA factible en el barrido: {mejor['mu']:.{decimales}f}")
        print(f"Alpha correspondiente              : {mejor['alpha']:.{decimales}f}")
    else:
        print("\nNo se encontraron soluciones MOMA factibles en el barrido.")
    return df


def graficar_crecimiento(resultados, ax=None):
    """Grafica los mu evaluados en MOMA y el máximo de SteadyCom."""
    df = resultados_a_dataframe(resultados)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(df["alpha"], df["mu"], linestyle="--", marker="o",
            label="Crecimiento evaluado en MOMA")
    factibles = df[df["factible"] == True]
    no_factibles = df[df["factible"] != True]
    if not factibles.empty:
        ax.scatter(factibles["alpha"], factibles["mu"],
                   label="MOMA factible", zorder=3)
    if not no_factibles.empty:
        ax.scatter(no_factibles["alpha"], no_factibles["mu"],
                   marker="x", label="Sin solución factible", zorder=3)

    ax.axhline(resultados["mu_steadycom"], linestyle=":",
               label="Mu máximo SteadyCom")
    ax.set_xlabel("Factor de crecimiento (alpha)")
    ax.set_ylabel("Tasa de crecimiento comunitario (mu)")
    ax.set_title("Crecimiento comunitario: SteadyCom vs MOMA")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax



def graficar_abundancias(resultados, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    colores = {
        "B1": "tab:blue",
        "B2": "tab:orange",
        "B3": "tab:green"
    }

    abundancias_ref = resultados["abundancias_steadycom"]
    # Curvas MOMA
    for organismo, color in colores.items():
        mus = []
        abundancias = []
        for fila in resultados["filas"]:
            abund = fila["resultado"].get("abundancias")
            if abund is not None and organismo in abund:
                mus.append(fila["alpha"] * resultados["mu_techo"])
                abundancias.append(abund[organismo])
        ax.plot(mus, abundancias, marker="o", linestyle="-", color=color, label=f"{organismo} - MOMA")

    # Referencias SteadyCom
    for organismo, color in colores.items():
        valor = abundancias_ref[organismo]
        ax.axhline(y=valor, color=color, linestyle="--", linewidth=1.8, alpha=0.85)

    # Leyenda explícita
    handles = []
    for organismo, color in colores.items():
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                marker="o",
                linestyle="-",
                label=f"{organismo} - MOMA"
            )
        )

    for organismo, color in colores.items():
        valor = abundancias_ref[organismo]
        etiqueta = f"{organismo} - SteadyCom ({valor:.2f})"
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle="--",
                label=etiqueta
            )
        )

    ax.set_title("Abundancias: SteadyCom vs MOMA")
    ax.set_xlabel("Tasa de crecimiento comunitario (mu)")
    ax.set_ylabel("Abundancia relativa")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, loc="best", fontsize=9)

    return fig, ax


def graficar_comparacion(resultados):
    """Genera y muestra los gráficos de crecimiento y abundancias."""
    fig_mu, ax_mu = graficar_crecimiento(resultados)
    fig_ab, ax_ab = graficar_abundancias(resultados)
    plt.show()
    return (fig_mu, ax_mu), (fig_ab, ax_ab)