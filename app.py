"""
Laboratorio de Redes: Optimalidad y Sink Tree (Johal, 2008)
Python + NetworkX + Streamlit + Plotly

Ejecutar:  python -m streamlit run app.py

Modelo:  G=(V,E) no dirigido. Origen H, destino B.
  - Johal (original):  minimizar H(P) = suma de h_ij, con h_ij = 1 (saltos)
  - Extension retardo: minimizar D(P) = suma de d_ij
  - Extension multicriterio (propia del laboratorio):
        C(P) = suma de [ alpha + beta*(d_ij/d_ref) + gamma*(b_ref/b_ij) ]
    con alpha+beta+gamma = 1, d_ref y b_ref = maximos positivos de la tabla actual.
  - Ancho de banda del camino: B(P) = min b_ij (cuello de botella; NO es aditivo).

Los valores en ms y Mbps y la funcion multicriterio son construcciones del
laboratorio; no son resultados experimentales de Johal.
"""

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ORIGEN, DESTINO = "H", "B"
NODOS = ["H", "D", "A", "B", "E", "J", "C", "O", "N"]

# Rutas preconfiguradas del articulo, en orden de prioridad (por saltos)
RUTAS_ESTATICAS = {
    "PSRC": ["H", "D", "A", "B"],
    "SSRC": ["H", "E", "J", "C", "B"],
    "TSRC": ["H", "O", "N", "J", "C", "B"],
}

# (origen, destino, retardo_ms, banda_Mbps, disponible)
# Nota: la guia solo indica B(P3)=50 Mbps; aqui se asumen 50 Mbps en H-O, O-N y N-J.
# Ajustar estos tres valores si la tabla oficial del laboratorio indica otros.
ENLACES_BASE = [
    ("H", "D", 8, 100, True),
    ("D", "A", 9, 100, True),
    ("A", "B", 7, 100, True),
    ("H", "E", 5, 50, True),
    ("E", "J", 10, 50, True),
    ("J", "C", 8, 100, True),
    ("C", "B", 6, 100, True),
    ("H", "O", 7, 50, True),
    ("O", "N", 9, 50, True),
    ("N", "J", 9, 50, True),
]

PESOS_POR_DEFECTO = (0.5, 0.35, 0.15)

MODOS = {
    "Johal: mínimo de saltos": "saltos",
    "Mínimo retardo": "retardo",
    "Multicriterio": "multicriterio",
}

POSICIONES = {
    "H": (0, 0), "D": (1, 1), "A": (2.5, 1), "B": (4, 0),
    "E": (1, 0), "J": (2, 0), "C": (3, 0),
    "O": (1, -1), "N": (2, -1),
}


# ---------------------------------------------------------------- modelo

def tabla_base() -> pd.DataFrame:
    return pd.DataFrame(
        ENLACES_BASE,
        columns=["Origen", "Destino", "Retardo_ms", "Banda_Mbps", "Disponible"],
    )


def ajustar(df, o, d, ret=None, banda=None, disp=None) -> pd.DataFrame:
    """Devuelve una copia de la tabla con el enlace o-d modificado."""
    df = df.copy()
    m = ((df.Origen == o) & (df.Destino == d)) | ((df.Origen == d) & (df.Destino == o))
    if ret is not None:
        df.loc[m, "Retardo_ms"] = ret
    if banda is not None:
        df.loc[m, "Banda_Mbps"] = banda
    if disp is not None:
        df.loc[m, "Disponible"] = disp
    return df


def normalizar(a, b, g):
    """Normaliza los pesos para que sumen 1. Si todos son 0 usa 1/3 cada uno."""
    s = a + b + g
    if s <= 0:
        return 1 / 3, 1 / 3, 1 / 3, True
    return a / s, b / s, g / s, False


def referencias(df):
    """d_ref y b_ref: maximos positivos registrados en la tabla de enlaces actual."""
    d = df.loc[df.Retardo_ms > 0, "Retardo_ms"]
    b = df.loc[df.Banda_Mbps > 0, "Banda_Mbps"]
    return (float(d.max()) if len(d) else 1.0), (float(b.max()) if len(b) else 1.0)


def enlace_valido(r) -> bool:
    return bool(r.Disponible) and r.Banda_Mbps > 0 and r.Retardo_ms >= 0


def metricas_ruta(G, ruta):
    """H(P), D(P), B(P), C(P) de una ruta; None si algun enlace no esta disponible."""
    saltos, ret, bandas, costo = 0, 0.0, [], 0.0
    for u, v in zip(ruta[:-1], ruta[1:]):
        if not G.has_edge(u, v):
            return None
        e = G[u][v]
        saltos += 1
        ret += e["d"]
        bandas.append(e["b"])
        costo += e["c"]
    return {"saltos": saltos, "retardo": ret, "banda": min(bandas), "costo": costo}


def evaluar(df, modo, pesos=PESOS_POR_DEFECTO):
    """Construye el grafo y calcula ruta estatica, ruta optima y Sink Tree."""
    a, b, g, _ = normalizar(*pesos)
    d_ref, b_ref = referencias(df)

    G = nx.Graph()
    G.add_nodes_from(NODOS)
    for r in df.itertuples(index=False):
        if not enlace_valido(r):
            continue
        c = a + b * r.Retardo_ms / d_ref + g * b_ref / r.Banda_Mbps
        w = {"saltos": 1.0, "retardo": float(r.Retardo_ms), "multicriterio": c}[modo]
        G.add_edge(r.Origen, r.Destino, d=float(r.Retardo_ms),
                   b=float(r.Banda_Mbps), c=c, w=w)

    # Seleccion estatica: primera ruta preconfigurada con todos sus enlaces activos
    estatica = None
    for nombre, ruta in RUTAS_ESTATICAS.items():
        if metricas_ruta(G, ruta) is not None:
            estatica = (nombre, ruta)
            break

    # Ruta optima recalculada (Dijkstra sobre pesos no negativos)
    try:
        optima = nx.dijkstra_path(G, ORIGEN, DESTINO, weight="w")
    except nx.NetworkXNoPath:
        optima = None

    # Sink Tree: un solo siguiente salto hacia B desde cada nodo alcanzable
    dist, caminos = nx.single_source_dijkstra(G, DESTINO, weight="w")
    siguiente = {v: p[-2] for v, p in caminos.items() if v != DESTINO}

    return {
        "G": G, "a": a, "b": b, "g": g, "d_ref": d_ref, "b_ref": b_ref,
        "estatica": estatica, "optima": optima,
        "dist": dist, "siguiente": siguiente,
        "metricas": {n: metricas_ruta(G, r) for n, r in RUTAS_ESTATICAS.items()},
    }


def tabla_resultados(res, modo) -> pd.DataFrame:
    filas = []
    rutas = dict(RUTAS_ESTATICAS)
    if res["optima"] and res["optima"] not in rutas.values():
        rutas["Óptima recalculada"] = res["optima"]
    for nombre, ruta in rutas.items():
        m = metricas_ruta(res["G"], ruta)
        filas.append({
            "Ruta": nombre,
            "Camino": "–".join(ruta),
            "Disponible": m is not None,
            "Saltos_H": m["saltos"] if m else None,
            "Retardo_D_ms": m["retardo"] if m else None,
            "Banda_B_Mbps": m["banda"] if m else None,
            "Costo_C": round(m["costo"], 4) if m else None,
            "Seleccion_estatica": bool(res["estatica"] and res["estatica"][0] == nombre),
            "Optima_recalculada": res["optima"] == ruta,
            "Funcion_objetivo": modo,
            "alpha": round(res["a"], 4), "beta": round(res["b"], 4),
            "gamma": round(res["g"], 4),
            "d_ref": res["d_ref"], "b_ref": res["b_ref"],
        })
    return pd.DataFrame(filas)


# ------------------------------------------------------ experimentos A-F

def _desc(res):
    partes = []
    for n, m in res["metricas"].items():
        if m is None:
            partes.append(f"{n}: no disponible")
        else:
            partes.append(f"{n}: H={m['saltos']}, D={m['retardo']:g} ms, "
                          f"B={m['banda']:g} Mbps, C={m['costo']:.3f}")
    est = res["estatica"][0] if res["estatica"] else "ninguna"
    opt = "–".join(res["optima"]) if res["optima"] else "sin ruta"
    return "; ".join(partes) + f" | Estática: {est} | Óptima: {opt}"


def resolver_experimentos() -> pd.DataFrame:
    base = tabla_base()
    filas = []

    filas.append(("A. Réplica original", "Saltos, tabla base",
                  _desc(evaluar(base, "saltos"))))

    filas.append(("B. Cambio de retardo", "H–D = 40 ms, mínimo retardo",
                  _desc(evaluar(ajustar(base, "H", "D", ret=40), "retardo"))))

    filas.append(("C. Fallo primario", "H–D caído, saltos",
                  _desc(evaluar(ajustar(base, "H", "D", disp=False), "saltos"))))
    filas.append(("C. Fallo primario", "H–D caído, retardo",
                  _desc(evaluar(ajustar(base, "H", "D", disp=False), "retardo"))))

    d2 = ajustar(ajustar(base, "H", "D", disp=False), "H", "E", disp=False)
    filas.append(("D. Doble fallo", "H–D y H–E caídos, saltos",
                  _desc(evaluar(d2, "saltos"))))

    e = ajustar(base, "E", "J", banda=5)
    filas.append(("E. Cuello de botella", "E–J = 5 Mbps, multicriterio (0.5, 0.35, 0.15)",
                  _desc(evaluar(e, "multicriterio", PESOS_POR_DEFECTO))))

    for pesos in [(1, 0, 0), (0, 1, 0), (0, 0, 1), PESOS_POR_DEFECTO]:
        filas.append(("F. Cambio de pesos", f"(α,β,γ) = {pesos}",
                      _desc(evaluar(base, "multicriterio", pesos))))

    return pd.DataFrame(filas, columns=["Prueba", "Configuración", "Resultado"])


# ---------------------------------------------------------------- figuras

def _trazo_aristas(df, ruta, color, ancho, dash=None, nombre=""):
    xs, ys = [], []
    for u, v in zip(ruta[:-1], ruta[1:]):
        xs += [POSICIONES[u][0], POSICIONES[v][0], None]
        ys += [POSICIONES[u][1], POSICIONES[v][1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", name=nombre,
                      line=dict(color=color, width=ancho, dash=dash), hoverinfo="skip")


def figura_grafo(df, res):
    fig = go.Figure()
    # Todas las aristas de la tabla
    for r in df.itertuples(index=False):
        activo = enlace_valido(r)
        fig.add_trace(_trazo_aristas(
            df, [r.Origen, r.Destino],
            "#9aa0a6" if activo else "#d93025", 2, None if activo else "dash"))
    if res["estatica"]:
        fig.add_trace(_trazo_aristas(df, res["estatica"][1], "#f9ab00", 8, "dot",
                                     f"Estática ({res['estatica'][0]})"))
    if res["optima"]:
        fig.add_trace(_trazo_aristas(df, res["optima"], "#1a73e8", 5, None,
                                     "Óptima recalculada"))
    # Etiquetas de aristas
    fig.add_trace(go.Scatter(
        x=[(POSICIONES[r.Origen][0] + POSICIONES[r.Destino][0]) / 2 for r in df.itertuples()],
        y=[(POSICIONES[r.Origen][1] + POSICIONES[r.Destino][1]) / 2 + 0.12 for r in df.itertuples()],
        mode="text", textfont=dict(size=11), showlegend=False, hoverinfo="skip",
        text=[f"{r.Retardo_ms:g} ms / {r.Banda_Mbps:g} Mbps" for r in df.itertuples()]))
    # Nodos
    fig.add_trace(go.Scatter(
        x=[POSICIONES[n][0] for n in NODOS], y=[POSICIONES[n][1] for n in NODOS],
        mode="markers+text", text=NODOS, textposition="middle center",
        marker=dict(size=34, color="#e8f0fe", line=dict(color="#1a73e8", width=2)),
        textfont=dict(size=15), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
                      legend=dict(orientation="h", y=-0.05))
    return fig


def figura_sink(res):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[POSICIONES[n][0] for n in NODOS], y=[POSICIONES[n][1] for n in NODOS],
        mode="markers+text", text=NODOS, textposition="middle center",
        marker=dict(size=34, color=["#fce8e6" if n == DESTINO else "#e8f0fe" for n in NODOS],
                    line=dict(color="#1a73e8", width=2)),
        textfont=dict(size=15), showlegend=False, hoverinfo="skip"))
    for v, n in res["siguiente"].items():
        fig.add_annotation(
            x=POSICIONES[n][0], y=POSICIONES[n][1],
            ax=POSICIONES[v][0], ay=POSICIONES[v][1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.5,
            arrowcolor="#188038", standoff=18, startstandoff=18)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    return fig


def mostrar(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # versiones antiguas de Streamlit
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------- app

def main():
    st.set_page_config(page_title="Sink Tree – Johal", layout="wide")
    st.title("Optimalidad y Sink Tree: rutas de H hacia B")
    st.caption("Basado en Johal (2008). Los ms/Mbps y la función multicriterio "
               "son construcciones del laboratorio, no datos del artículo.")

    if "version" not in st.session_state:
        st.session_state.version = 0

    # --- Barra lateral
    st.sidebar.header("Parámetros")
    if st.sidebar.button("Restablecer tabla"):
        st.session_state.version += 1
    modo_txt = st.sidebar.radio("Función objetivo", list(MODOS))
    modo = MODOS[modo_txt]
    st.sidebar.markdown("**Pesos multicriterio** (se normalizan)")
    a_in = st.sidebar.slider("α  saltos", 0.0, 1.0, PESOS_POR_DEFECTO[0], 0.05)
    b_in = st.sidebar.slider("β  retardo", 0.0, 1.0, PESOS_POR_DEFECTO[1], 0.05)
    g_in = st.sidebar.slider("γ  baja capacidad", 0.0, 1.0, PESOS_POR_DEFECTO[2], 0.05)
    a, b, g, todos_cero = normalizar(a_in, b_in, g_in)
    if todos_cero:
        st.sidebar.warning("Los tres pesos son 0: se usa 1/3 para cada uno.")
    st.sidebar.write(f"Normalizados: α={a:.3f}, β={b:.3f}, γ={g:.3f}")

    # --- Sección 1: tabla editable
    st.header("1. Tabla de enlaces")
    df = st.data_editor(
        tabla_base(), key=f"editor_{st.session_state.version}",
        disabled=["Origen", "Destino"], hide_index=True,
        column_config={
            "Retardo_ms": st.column_config.NumberColumn("Retardo (ms)", min_value=0.0),
            "Banda_Mbps": st.column_config.NumberColumn("Banda (Mbps)", min_value=0.0),
            "Disponible": st.column_config.CheckboxColumn("Disponible"),
        })
    df = df.fillna({"Retardo_ms": 0, "Banda_Mbps": 0, "Disponible": False})
    invalidos = df[df.Disponible & (df.Banda_Mbps <= 0)]
    if len(invalidos):
        st.warning("Hay enlaces disponibles con banda 0: se ignoran, porque "
                   "b_ref/b_ij no está definido.")

    res = evaluar(df, modo, (a_in, b_in, g_in))

    # --- Sección 2: modelo y rutas
    st.header("2. Fórmulas y camino óptimo")
    st.latex(r"H(P)=\sum_{(i,j)\in P}h_{ij},\ h_{ij}=1 \qquad D(P)=\sum_{(i,j)\in P}d_{ij}")
    st.latex(r"C(P)=\sum_{(i,j)\in P}\left[\alpha+\beta\frac{d_{ij}}{d_{ref}}"
             r"+\gamma\frac{b_{ref}}{b_{ij}}\right],\quad \alpha+\beta+\gamma=1")
    st.latex(r"B(P)=\min_{(i,j)\in P} b_{ij}")
    st.write(f"d_ref = {res['d_ref']:g} ms · b_ref = {res['b_ref']:g} Mbps "
             "(registrarlos al comparar experimentos).")

    col1, col2 = st.columns(2)
    col1.metric("Selección estática (artículo)",
                "–".join(res["estatica"][1]) if res["estatica"] else "Sin ruta")
    col2.metric("Óptima recalculada",
                "–".join(res["optima"]) if res["optima"] else "Sin ruta")
    if res["estatica"] and res["optima"] and res["estatica"][1] != res["optima"]:
        st.info("La ruta estática y la recalculada difieren.")
    mostrar(figura_grafo(df, res))
    resultados = tabla_resultados(res, modo)
    st.dataframe(resultados, hide_index=True)

    # --- Sección 3: Sink Tree
    st.header("3. Sink Tree hacia B")
    st.write("Cada flecha indica el único siguiente salto desde ese router hacia B.")
    mostrar(figura_sink(res))
    st.dataframe(pd.DataFrame(
        [{"Router": v, "Siguiente_salto": n, "Costo_hasta_B": round(res["dist"][v], 3)}
         for v, n in sorted(res["siguiente"].items())]), hide_index=True)
    st.caption("Si hay empate entre dos caminos, NetworkX elige uno de forma determinista; "
               "el empate es un caso a mencionar en el análisis.")

    # --- Sección 4: definiciones, experimentos y exportación
    st.header("4. Resultados y exportación")
    with st.expander("Solución de los experimentos A–F (tabla base)"):
        st.dataframe(resolver_experimentos(), hide_index=True)
    st.markdown(
        "- **Principio de optimalidad:** con costos aditivos no negativos, todo "
        "subcamino de una ruta óptima hacia B también es óptimo.\n"
        "- **B(P)=min b_ij** es un cuello de botella, no una suma, y por eso no se "
        "optimiza directamente con Dijkstra; se usa la penalización b_ref/b_ij.")
    st.download_button("Descargar CSV de resultados",
                       resultados.to_csv(index=False).encode("utf-8"),
                       "resultados_sink_tree.csv", "text/csv")


if __name__ == "__main__":
    main()
