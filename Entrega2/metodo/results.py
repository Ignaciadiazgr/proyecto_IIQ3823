
"""
Funciones para organizar, imprimir y graficar la comparación
SteadyCom vs Community MOMA.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from metodo.steadycom_adapted import _steadycom_mu_fijo


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



def graficar_abundancias(resultados, medio=None, ax=None):
    """
    Grafica abundancias MOMA y SteadyCom.

    CASO 1: Existen residentes.
        Calcula SteadyCom exclusivamente para el consorcio
        residente, fijando cada tasa de crecimiento usada
        en el barrido de MOMA.

    CASO 2: No existen residentes.
        Grafica las abundancias de una única referencia
        SteadyCom de la comunidad completa como líneas
        horizontales punteadas.

    Las curvas MOMA se mantienen como líneas continuas.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    residentes = list(resultados.get("residentes", []))
    invasores = list(resultados.get("invasores", []))

    organismos = _obtener_organismos(resultados)

    if not organismos:
        raise ValueError("No hay organismos para graficar.")

    colores = {
        org: f"C{i % 10}"
        for i, org in enumerate(organismos)
    }

    # ========================================================
    # 1. CURVAS MOMA
    # ========================================================

    filas = [
        fila
        for fila in resultados.get("filas", [])
        if fila.get("factible", False)
        and fila.get("abundancias") is not None
    ]

    filas.sort(key=lambda fila: fila["mu"])

    for org in organismos:

        datos = [
            (fila["mu"], fila["abundancias"][org])
            for fila in filas
            if org in fila["abundancias"]
        ]

        if not datos:
            continue

        mus, abundancias = zip(*datos)

        ax.plot(
            mus,
            abundancias,
            marker="o",
            linestyle="-",
            linewidth=2,
            color=colores[org],
            label=f"{org} - MOMA"
        )

    # ========================================================
    # 2. REFERENCIAS STEADYCOM
    # ========================================================

    if residentes:

        # ----------------------------------------------------
        # CASO 1: STEADYCOM DEL CONSORCIO RESIDENTE
        # ----------------------------------------------------

        if medio is None:
            raise ValueError(
                "Para graficar SteadyCom a crecimiento fijo "
                "debes proporcionar medio=medio."
            )

        referencia_residentes = resultados.get(
            "steadycom_residentes"
        )

        if referencia_residentes is not None:

            comunidad_residente = referencia_residentes[
                "comunidad"
            ]

            mu_max_residentes = referencia_residentes[
                "mu_max"
            ]

        else:

            # Reconstruir la comunidad residente cuando
            # el resultado todavía no contiene esa referencia.
            referencias = resultados["referencias"]

            modelos_originales = referencias.get("modelos")

            if modelos_originales is None:
                # Compatibilidad con el código previo.
                modelos_originales = {
                    org: ref["modelo"]
                    for org, ref in referencias["pfba"].items()
                }

            modelos_residentes = {
                org: modelos_originales[org]
                for org in residentes
            }

            referencia_residentes = (
                calcular_referencia_comunitaria(
                    modelos_residentes,
                    medio
                )
            )

            comunidad_residente = referencia_residentes[
                "comunidad"
            ]

            mu_max_residentes = referencia_residentes[
                "mu_max"
            ]

        # Valores de crecimiento de las soluciones MOMA.
        mus = sorted({
            float(fila["mu"])
            for fila in resultados.get("filas", [])
        })

        curvas = {
            org: []
            for org in residentes
        }

        # Resolver SteadyCom a crecimiento fijo
        # para cada tasa del barrido MOMA.
        for mu in mus:

            # Crecimiento mayor al máximo residente:
            # no existe solución de referencia.
            if mu > mu_max_residentes + 1e-6:
                abundancias_sc = None
            else:
                abundancias_sc = _steadycom_mu_fijo(
                    comunidad_residente,
                    medio,
                    mu
                )

            for org in residentes:

                if abundancias_sc is None:
                    curvas[org].append(np.nan)
                else:
                    curvas[org].append(
                        abundancias_sc[org]
                    )

        # Graficar solo los residentes.
        for org in residentes:

            ax.plot(
                mus,
                curvas[org],
                linestyle="--",
                marker="s",
                linewidth=1.8,
                color=colores[org],
                alpha=0.85,
                label=f"{org} - SteadyCom residentes"
            )

        ax.set_title(
            "Abundancias: MOMA vs SteadyCom residentes"
        )

    else:

        # ----------------------------------------------------
        # CASO 2: SOLO INVASORES
        # ----------------------------------------------------
        # Se calcula SteadyCom de la comunidad completa
        # para cada valor de mu del barrido MOMA.
        # Esto genera una única referencia comunitaria,
        # pero con abundancias de B1, B2 y B3 que pueden
        # variar con mu.

        if medio is None:
            raise ValueError(
                "Para graficar SteadyCom completo a crecimiento fijo "
                "debes proporcionar medio=medio."
            )

        referencia_completa = resultados["referencias"]["steadycom"]
        comunidad_completa = referencia_completa["comunidad"]
        mu_max_completo = referencia_completa["mu_max"]

        mus = sorted({
            float(fila["mu"])
            for fila in resultados.get("filas", [])
            if fila.get("mu") is not None
        })

        curvas = {
            org: []
            for org in invasores
        }

        for mu in mus:

            if mu > mu_max_completo + 1e-6:
                abundancias_sc = None
            else:
                abundancias_sc = _steadycom_mu_fijo(
                    comunidad_completa,
                    medio,
                    mu
                )

            for org in invasores:
                if abundancias_sc is None:
                    curvas[org].append(np.nan)
                else:
                    curvas[org].append(abundancias_sc[org])

        # Graficar la referencia comunitaria completa
        # como curvas punteadas
        for org in invasores:
            ax.plot(
                mus,
                curvas[org],
                linestyle="--",
                marker="s",
                linewidth=1.8,
                color=colores[org],
                alpha=0.85,
                label=f"{org} - SteadyCom completo"
            )

        ax.set_title(
            "Abundancias: MOMA vs SteadyCom completo"
        )

    # ========================================================
    # 3. FORMATO
    # ========================================================

    ax.set_xlabel(
        r"Tasa de crecimiento comunitario ($\mu$)"
    )

    ax.set_ylabel("Abundancia relativa")
    ax.set_ylim(-0.02, 1.05)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

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
