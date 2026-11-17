
"""
Funciones para organizar, imprimir y graficar la comparación
SteadyCom vs Community MOMA.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _obtener_organismos(resultados):
    """
    Obtiene dinámicamente los organismos considerados
    en el análisis, incluyendo residentes e invasores.
    """

    organismos = []

    # Residentes e invasores
    for grupo in ("residentes", "invasores"):
        for org in resultados.get(grupo, []):
            if org not in organismos:
                organismos.append(org)

    # Abundancias de referencia
    for clave in (
        "abundancias_steadycom",
        "abundancias_referencia_residentes"
    ):
        for org in resultados.get(clave, {}):
            if org not in organismos:
                organismos.append(org)

    # Abundancias obtenidas por MOMA
    for fila in resultados.get("filas", []):
        abundancias = fila.get("abundancias") or {}

        for org in abundancias:
            if org not in organismos:
                organismos.append(org)

    return organismos


def resultados_a_dataframe(resultados):
    """
    Genera un DataFrame con una fila por alpha.

    Incluye:
        - Alpha y crecimiento comunitario.
        - Estado y factibilidad.
        - Función objetivo y tiempo de resolución.
        - Costos metabólicos.
        - Abundancias MOMA y referencias.
    """

    organismos = _obtener_organismos(resultados)

    abund_sc = resultados.get("abundancias_steadycom", {})
    abund_ref = resultados.get(
        "abundancias_referencia_residentes", {}
    )

    registros = []

    for fila in resultados.get("filas", []):

        registro = {
            "alpha": fila.get("alpha"),
            "mu": fila.get("mu"),
            "status": fila.get("status"),
            "factible": fila.get("factible", False),
            "objetivo": fila.get("objetivo"),
            "cost_residentes": fila.get("cost_residentes"),
            "cost_invasores": fila.get("cost_invasores"),
            "cost_total": fila.get("cost_total"),
            "tiempo": fila.get("tiempo")
        }

        abund_moma = fila.get("abundancias") or {}

        for org in organismos:

            registro[f"{org}_MOMA"] = abund_moma.get(org)

            # Se prioriza la referencia de residentes si existe
            if org in abund_ref:
                registro[f"{org}_Referencia"] = abund_ref[org]
            else:
                registro[f"{org}_Referencia"] = abund_sc.get(org)

        registros.append(registro)

    return pd.DataFrame(registros)


def imprimir_resumen(resultados, decimales=6):
    """
    Imprime un resumen del barrido y devuelve el DataFrame.
    """

    print("=" * 65)
    print("COMPARACIÓN STEADYCOM vs COMMUNITY MOMA")
    print("=" * 65)

    mu_steadycom = resultados.get("mu_steadycom")
    mu_techo = resultados.get("mu_techo")

    if mu_steadycom is not None:
        print(
            f"Mu máximo SteadyCom reportado : "
            f"{mu_steadycom:.{decimales}f}"
        )

    if mu_techo is not None:
        print(
            f"Mu techo con coexistencia     : "
            f"{mu_techo:.{decimales}f}"
        )

    print("\nAbundancias de referencia SteadyCom:")

    abund_sc = resultados.get("abundancias_steadycom", {})

    if abund_sc:
        for org, x in abund_sc.items():
            print(f"  {org}: {x:.{decimales}f}")
    else:
        print("  No hay abundancias de referencia SteadyCom.")

    print("\nAbundancias de referencia de residentes:")

    abund_ref = resultados.get(
        "abundancias_referencia_residentes", {}
    )

    if abund_ref:
        for org, x in abund_ref.items():
            print(f"  {org}: {x:.{decimales}f}")
    else:
        print("  No hay residentes con abundancias de referencia.")

    df = resultados_a_dataframe(resultados)

    if df.empty:
        print("\nNo hay resultados disponibles para el barrido.")
        return df

    factibles = df[df["factible"] == True]

    print("\nResumen de resoluciones:")
    print(f"  Resoluciones ejecutadas : {len(df)}")
    print(f"  Soluciones factibles    : {len(factibles)}")
    print(f"  Sin solución factible   : {len(df) - len(factibles)}")

    if not factibles.empty:

        mejor = factibles.loc[factibles["mu"].idxmax()]

        print(
            f"\nMayor mu MOMA factible en el barrido: "
            f"{mejor['mu']:.{decimales}f}"
        )

        print(
            f"Alpha correspondiente              : "
            f"{mejor['alpha']:.{decimales}f}"
        )

        if pd.notna(mejor["cost_total"]):
            print(
                f"Costo total correspondiente         : "
                f"{mejor['cost_total']:.{decimales}f}"
            )

        print(
            f"Estado de optimización               : "
            f"{mejor['status']}"
        )

    else:
        print("\nNo se encontraron soluciones MOMA factibles en el barrido.")

    return df


def graficar_abundancias(resultados, ax=None):
    """
    Grafica las abundancias MOMA y las compara con las
    abundancias de referencia disponibles.

    Los organismos se detectan automáticamente.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    organismos = _obtener_organismos(resultados)

    if not organismos:
        raise ValueError(
            "No hay organismos disponibles para graficar."
        )

    abund_sc = resultados.get("abundancias_steadycom", {})
    abund_ref = resultados.get(
        "abundancias_referencia_residentes", {}
    )

    colores = {
        org: f"C{i % 10}"
        for i, org in enumerate(organismos)
    }

    handles = []

    # Curvas MOMA: solamente soluciones factibles
    filas_factibles = [
        fila
        for fila in resultados.get("filas", [])
        if fila.get("factible", False)
        and fila.get("abundancias") is not None
    ]

    for organismo in organismos:

        datos = [
            (fila["mu"], fila["abundancias"][organismo])
            for fila in filas_factibles
            if organismo in fila["abundancias"]
        ]

        if datos:

            datos.sort(key=lambda x: x[0])

            mus = [dato[0] for dato in datos]
            abundancias = [dato[1] for dato in datos]

            ax.plot(
                mus,
                abundancias,
                marker="o",
                linestyle="-",
                color=colores[organismo],
                label=f"{organismo} - MOMA"
            )

            handles.append(
                Line2D(
                    [0], [0],
                    color=colores[organismo],
                    marker="o",
                    linestyle="-",
                    label=f"{organismo} - MOMA"
                )
            )

    # Referencias de abundancia
    referencias = {}

    referencias.update(abund_sc)
    referencias.update(abund_ref)

    for organismo, valor in referencias.items():

        color = colores.get(organismo)

        if color is None or valor is None:
            continue

        ax.axhline(
            y=valor,
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.85
        )

        handles.append(
            Line2D(
                [0], [0],
                color=color,
                linestyle="--",
                label=f"{organismo} - Referencia ({valor:.2f})"
            )
        )

    ax.set_title("Abundancias: referencias vs Community MOMA")
    ax.set_xlabel(r"Tasa de crecimiento comunitario ($\mu$)")
    ax.set_ylabel("Abundancia relativa")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)

    if handles:
        ax.legend(handles=handles, loc="best", fontsize=9)

    fig.tight_layout()

    return fig, ax


def graficar_costo_metabolico(resultados, ax=None):
    """
    Grafica los costos metabólicos de residentes e invasores
    y el costo total en función del crecimiento comunitario.

    Si no existen residentes, su costo se considera cero.
    """

    filas = resultados.get("filas", [])

    # Extraer únicamente soluciones factibles
    datos = [
        fila
        for fila in filas
        if fila.get("factible", False)
        and fila.get("objetivo") is not None
    ]

    if not datos:
        raise ValueError(
            "No hay soluciones factibles para graficar."
        )

    datos.sort(key=lambda fila: fila["mu"])

    mu = [fila["mu"] for fila in datos]

    costo_residentes = [
        fila.get("cost_residentes")
        if fila.get("cost_residentes") is not None
        else 0.0
        for fila in datos
    ]

    costo_invasores = [
        fila.get("cost_invasores")
        if fila.get("cost_invasores") is not None
        else 0.0
        for fila in datos
    ]

    costo_total = [
        fila.get("cost_total")
        if fila.get("cost_total") is not None
        else (
            (fila.get("cost_residentes") or 0.0)
            + (fila.get("cost_invasores") or 0.0)
        )
        for fila in datos
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    ax.plot(
        mu,
        costo_residentes,
        marker="o",
        linewidth=2,
        label="Residentes"
    )

    ax.plot(
        mu,
        costo_invasores,
        marker="s",
        linewidth=2,
        label="Invasores"
    )

    ax.plot(
        mu,
        costo_total,
        marker="^",
        linewidth=2,
        linestyle="--",
        label="Costo total"
    )

    ax.set_xlabel(r"Tasa de crecimiento comunitario ($\mu$)")
    ax.set_ylabel("Costo metabólico (función objetivo)")
    ax.set_title("Costo metabólico vs. crecimiento comunitario")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig, ax
