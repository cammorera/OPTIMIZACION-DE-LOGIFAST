"""
app.py — Cross Docking Optimizer · LogiFast CR · UCR I-2026
===========================================================
Streamlit app: carga datos formato TS5, resuelve MIP,
muestra resultados, Gantt y flujo de productos.
 
Dependencias: streamlit, pandas, plotly, pulp
(amplpy + HiGHS/CPLEX/Gurobi opcionales — mejora performance)
"""
 
import io
from datetime import datetime
 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
 
from solver import (
    CrossDockInstance,
    SolverResult,
    parse_ts5,
    solve,
)
 
# ─── Datos de ejemplo (instancia TS5) ────────────────────────────────────────
_TS5_EXAMPLE = """\
i\t5
o\t3
n\t8
r\t1\t1\t170
r\t2\t1\t6
r\t2\t2\t6
r\t2\t3\t19
r\t2\t4\t50
r\t2\t5\t38
r\t2\t6\t6
r\t2\t7\t19
r\t2\t8\t56
r\t3\t1\t49
r\t3\t2\t31
r\t3\t3\t60
r\t3\t6\t12
r\t3\t7\t37
r\t3\t8\t31
r\t4\t5\t143
r\t4\t7\t47
r\t5\t4\t58
r\t5\t5\t36
r\t5\t7\t72
r\t5\t8\t14
s\t1\t1\t75
s\t1\t2\t12
s\t1\t3\t59
s\t1\t6\t9
s\t1\t7\t98
s\t1\t8\t40
s\t2\t1\t150
s\t2\t5\t217
s\t3\t2\t25
s\t3\t3\t20
s\t3\t4\t108
s\t3\t6\t9
s\t3\t7\t77
s\t3\t8\t61
"""
 
# ─── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cross Docking Optimizer · LogiFast CR",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;500&display=swap');
 
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
 
/* Fondo blanco general */
.stApp, .main .block-container {
    background-color: #ffffff !important;
}
 
/* KPI cards con fondo azul muy suave */
.kpi {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 1px 4px rgba(59,130,246,0.08);
}
.kpi-val   { font-size: 2.2rem; font-weight: 700; color: #1d4ed8; line-height: 1.1; }
.kpi-label { font-size: 0.78rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
 
/* Encabezados de sección */
.sec { font-size: 1rem; font-weight: 700; color: #1e3a8a;
       border-left: 4px solid #2563eb; padding-left: 10px;
       margin: 1.6rem 0 .7rem 0; background: #f0f7ff;
       border-radius: 0 6px 6px 0; padding-top: 6px; padding-bottom: 6px; }
 
/* Badges */
.badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 50%;
    font-weight: 700; font-size: .9rem; margin: 3px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.12);
}
.bi { background: #2563eb; color: #fff; }
.bo { background: #0891b2; color: #fff; }
 
/* Sidebar en azul oscuro */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a8a 0%, #1d4ed8 100%) !important;
}
div[data-testid="stSidebar"] * {
    color: #e0eaff !important;
}
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label,
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span {
    color: #bfdbfe !important;
}
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
div[data-testid="stSidebar"] .stDivider { border-color: #3b82f6 !important; }
 
/* Botones primarios en azul */
div.stButton > button[kind="primary"],
div.stButton > button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: 0 2px 6px rgba(37,99,235,0.25) !important;
}
div.stButton > button:hover {
    background-color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
}
 
/* Botones de descarga */
div.stDownloadButton > button {
    background-color: #eff6ff !important;
    color: #1d4ed8 !important;
    border: 2px solid #2563eb !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
div.stDownloadButton > button:hover {
    background-color: #2563eb !important;
    color: #ffffff !important;
}
 
/* Tabs estilo azul */
div[data-testid="stTabs"] button {
    color: #64748b !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}
 
/* Dataframes con header azul */
.stDataFrame thead tr th {
    background-color: #2563eb !important;
    color: white !important;
}
 
/* Título principal */
h1 { color: #1e3a8a !important; }
h2, h3 { color: #1d4ed8 !important; }
</style>
""", unsafe_allow_html=True)
 
 
# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚛 Cross Docking")
    st.markdown("**LogiFast CR — UCR · IE**")
    st.markdown("*MIP · I Semestre 2026*")
    st.divider()
 
    st.markdown("### ⚙️ Solver")
    solver_choice = st.selectbox(
        "Solver preferido",
        ["highs", "cplex", "gurobi", "cbc"],
        help="Si amplpy no está instalado se usa PuLP/CBC automáticamente.",
    )
    time_limit = st.slider("Tiempo límite (s)", 30, 600, 300, 30)
    mip_gap    = st.select_slider(
        "MIP Gap", [0.0001, 0.001, 0.005, 0.01, 0.05], value=0.001
    )
 
    st.divider()
    st.markdown("### 📋 Parámetros operativos")
    st.code("t_unit   = 1  min/ud\nt_trans  = 5  min/lote\nt_switch = 10 min/cambio",
            language="text")
    st.divider()
    st.caption("Universidad de Costa Rica · Ing. Industrial")
 
 
# ─── Encabezado ───────────────────────────────────────────────────────────────
st.markdown("# 🚛 Cross Docking Optimizer")
st.markdown("**Minimización del makespan · Programación Entera Mixta (MIP)**")
st.divider()
 
# ─── Carga de datos ───────────────────────────────────────────────────────────
st.markdown('<div class="sec">📂 Datos de entrada</div>', unsafe_allow_html=True)
 
tab_ex, tab_up = st.tabs(["🗂 Ejemplo TS5", "📤 Subir archivo"])
 
inst: CrossDockInstance | None = None
 
with tab_ex:
    st.code(_TS5_EXAMPLE, language="text")
    if st.button("✅ Usar datos de ejemplo", use_container_width=True):
        inst = parse_ts5(_TS5_EXAMPLE)
        st.session_state["inst"] = inst
 
with tab_up:
    uploaded = st.file_uploader("Archivo .txt con formato TS5", type=["txt"])
    if uploaded:
        content = uploaded.read().decode("utf-8")
        inst = parse_ts5(content)
        st.session_state["inst"] = inst
        st.success(f"'{uploaded.name}' cargado.")
 
# Recuperar de sesión
if inst is None:
    inst = st.session_state.get("inst")
 
# ─── Vista previa de datos ────────────────────────────────────────────────────
if inst is not None:
    errors = inst.validate_balance()
    if errors:
        st.error("⚠ Desequilibrio oferta-demanda:\n" + "\n".join(errors))
        st.stop()
 
    st.success(
        f"✅ Instancia cargada — "
        f"{inst.num_inbound} camiones entrada | "
        f"{inst.num_outbound} camiones salida | "
        f"{inst.num_products} productos"
    )
 
    c1, c2 = st.columns(2)
 
    with c1:
        st.markdown('<div class="sec">📥 Camiones de entrada</div>', unsafe_allow_html=True)
        rows = []
        for i in inst.inbound_trucks():
            row = {"Camión": f"E{i}"}
            tot = 0
            for k in inst.products():
                q = inst.ri.get((i, k), 0)
                row[f"P{k}"] = q or ""
                tot += q
            row["Total"] = tot
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Camión"),
                     use_container_width=True, height=230)
 
    with c2:
        st.markdown('<div class="sec">📤 Camiones de salida</div>', unsafe_allow_html=True)
        rows = []
        for j in inst.outbound_trucks():
            row = {"Camión": f"S{j}"}
            tot = 0
            for k in inst.products():
                q = inst.sj.get((j, k), 0)
                row[f"P{k}"] = q or ""
                tot += q
            row["Total"] = tot
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Camión"),
                     use_container_width=True, height=200)
 
    # ─── Botón de optimización ────────────────────────────────────────────────
    st.divider()
    if st.button("🔍 Optimizar secuencia", type="primary", use_container_width=True):
        with st.spinner("Resolviendo MIP..."):
            result: SolverResult = solve(
                inst,
                preferred_solver=solver_choice,
                time_limit=time_limit,
                mip_gap=mip_gap,
            )
        st.session_state["result"] = result
 
    # ─── Resultados ───────────────────────────────────────────────────────────
    if "result" in st.session_state:
        result: SolverResult = st.session_state["result"]
 
        if result.status != "optimal":
            st.error(f"❌ No se encontró solución: {result.message or result.status}")
            st.stop()
 
        # KPIs
        st.markdown('<div class="sec">📊 Resultados</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        direct_count = sum(1 for val in result.v.values() if val)
        for col, val, label in [
            (k1, f"{result.makespan:.0f} min", "Makespan"),
            (k2, f"{result.solve_time:.1f} s",  "Tiempo cómputo"),
            (k3, str(direct_count),              "Transf. directas"),
            (k4, result.solver_used.split("/")[0], "Solver"),
        ]:
            col.markdown(
                f'<div class="kpi"><div class="kpi-val">{val}</div>'
                f'<div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True,
            )
 
        st.markdown("")
 
        # Órdenes
        ci, co = st.columns(2)
 
        with ci:
            st.markdown("**Orden camiones de ENTRADA**")
            st.markdown(
                " ".join(f'<span class="badge bi">{i}</span>'
                         for i in result.inbound_order),
                unsafe_allow_html=True,
            )
            rows = [
                {
                    "Pos": p,
                    "Camión": f"E{i}",
                    "Inicio": f"{result.a[i]:.1f}",
                    "Fin":    f"{result.a[i] + result.di[i]:.1f}",
                    "Duración": f"{result.di[i]:.0f} min",
                }
                for p, i in enumerate(result.inbound_order, 1)
            ]
            st.dataframe(pd.DataFrame(rows).set_index("Pos"),
                         use_container_width=True, height=230)
 
        with co:
            st.markdown("**Orden camiones de SALIDA**")
            st.markdown(
                " ".join(f'<span class="badge bo">{j}</span>'
                         for j in result.outbound_order),
                unsafe_allow_html=True,
            )
            rows = [
                {
                    "Pos": p,
                    "Camión": f"S{j}",
                    "Inicio": f"{result.d[j]:.1f}",
                    "Fin":    f"{result.d[j] + result.lj[j]:.1f}",
                    "Duración": f"{result.lj[j]:.0f} min",
                }
                for p, j in enumerate(result.outbound_order, 1)
            ]
            st.dataframe(pd.DataFrame(rows).set_index("Pos"),
                         use_container_width=True, height=200)
 
        # ── Gantt ─────────────────────────────────────────────────────────────
        st.markdown('<div class="sec">📅 Diagrama de Gantt</div>', unsafe_allow_html=True)
 
        fig = go.Figure()
        added: set = set()
 
        def _bar(task, start, duration, color, tipo):
            show = tipo not in added
            added.add(tipo)
            fig.add_trace(go.Bar(
                x=[duration], y=[task], base=[start],
                orientation="h",
                marker_color=color,
                name=tipo,
                showlegend=show,
                hovertemplate=(
                    f"<b>{task}</b><br>"
                    f"Inicio: {start:.1f} min<br>"
                    f"Fin: {start+duration:.1f} min<br>"
                    f"Duración: {duration:.0f} min<extra></extra>"
                ),
            ))
 
        for i in inst.inbound_trucks():
            _bar(f"Entrada {i}", result.a[i], result.di[i], "#2563eb", "Descarga")
        for j in inst.outbound_trucks():
            _bar(f"Salida {j}", result.d[j], result.lj[j], "#0891b2", "Carga")
 
        fig.add_vline(
            x=result.makespan, line_dash="dash", line_color="#dc2626", line_width=2,
            annotation_text=f"Makespan: {result.makespan:.0f} min",
            annotation_font_color="#dc2626",
        )
        fig.update_layout(
            barmode="stack",
            xaxis_title="Tiempo (minutos)",
            plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff",
            font_color="#1e3a8a",
            xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#cbd5e1"),
            yaxis=dict(gridcolor="#e2e8f0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#bfdbfe", borderwidth=1),
            height=max(320, 55*(inst.num_inbound + inst.num_outbound) + 100),
            margin=dict(l=20, r=20, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
 
        # ── Flujo de productos ────────────────────────────────────────────────
        st.markdown('<div class="sec">🔄 Flujo de productos</div>', unsafe_allow_html=True)
 
        flow_rows = [
            {
                "Entrada": f"E{i}",
                "Salida":  f"S{j}",
                "Producto": f"P{k}",
                "Unidades": qty,
                "Vía": "✅ Directa" if result.v.get((i,j), 0) else "🏭 Almacén temp.",
            }
            for (i, j, k), qty in sorted(result.x.items())
        ]
 
        if flow_rows:
            df_flow = pd.DataFrame(flow_rows)
            st.dataframe(df_flow, use_container_width=True, height=320)
 
            pivot = (
                df_flow.groupby(["Entrada","Salida"])["Unidades"]
                .sum().unstack(fill_value=0)
            )
            fig_h = px.imshow(
                pivot, text_auto=True,
                color_continuous_scale="Blues",
                title="Unidades transferidas E_i → S_j",
                labels=dict(x="Camión salida", y="Camión entrada", color="Unidades"),
            )
            fig_h.update_layout(
                plot_bgcolor="#f8fafc", paper_bgcolor="#ffffff", font_color="#1e3a8a"
            )
            st.plotly_chart(fig_h, use_container_width=True)
 
        # ── Exportar ──────────────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="sec">💾 Exportar</div>', unsafe_allow_html=True)
 
        def _summary_txt(inst: CrossDockInstance, r: SolverResult) -> str:
            lines = [
                "=" * 54,
                "  RESULTADOS — Cross Docking LogiFast CR",
                f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 54,
                f"  Solver   : {r.solver_used}",
                f"  Makespan : {r.makespan:.1f} minutos",
                f"  Cómputo  : {r.solve_time:.2f} s",
                "",
                "  ORDEN CAMIONES ENTRADA:",
            ]
            for p, i in enumerate(r.inbound_order, 1):
                lines.append(
                    f"    {p}. E{i}  inicio={r.a[i]:.1f}  "
                    f"fin={r.a[i]+r.di[i]:.1f}  dur={r.di[i]:.0f} min"
                )
            lines += ["", "  ORDEN CAMIONES SALIDA:"]
            for p, j in enumerate(r.outbound_order, 1):
                lines.append(
                    f"    {p}. S{j}  inicio={r.d[j]:.1f}  "
                    f"fin={r.d[j]+r.lj[j]:.1f}  dur={r.lj[j]:.0f} min"
                )
            lines += ["", "  FLUJO DE PRODUCTOS:"]
            for (i, j, k), qty in sorted(r.x.items()):
                via = "directa" if r.v.get((i,j), 0) else "almacén"
                lines.append(f"    E{i}→S{j}  P{k}: {qty} uds  [{via}]")
            lines.append("=" * 54)
            return "\n".join(lines)
 
        dc1, dc2 = st.columns(2)
        with dc1:
            st.download_button(
                "📄 Resumen (.txt)", data=_summary_txt(inst, result),
                file_name="crossdock_resultado.txt", mime="text/plain",
                use_container_width=True,
            )
        with dc2:
            if flow_rows:
                buf = io.StringIO()
                pd.DataFrame(flow_rows).to_csv(buf, index=False)
                st.download_button(
                    "📊 Flujo (.csv)", data=buf.getvalue(),
                    file_name="crossdock_flujo.csv", mime="text/csv",
                    use_container_width=True,
                )
 
else:
    st.info("👈 Selecciona los datos de ejemplo o sube tu archivo TS5 para comenzar.")
