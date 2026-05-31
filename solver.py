"""
solver.py — Cross Docking MIP · LogiFast CR · UCR I-2026
=========================================================
Incluye parser de formato TS5 + modelo MIP completo.
Usa amplpy (HiGHS/CPLEX/Gurobi) si está disponible,
cae a PuLP/CBC automáticamente si no hay licencia.

Parámetros operativos (fijos según el caso):
  t_unit   = 1  min/unidad  (carga o descarga)
  t_trans  = 5  min/lote    (traslado interno)
  t_switch = 10 min         (cambio entre camiones en muelle)

Variables de decisión:
  x[i,j,k]  : enteras >= 0   — unidades de producto k de camión entrada i a salida j
  v[i,j]    : binaria         — 1 si hay transferencia directa i → j
  alpha[i,i']: binaria        — 1 si camión entrada i precede a i'
  beta[j,j'] : binaria        — 1 si camión salida j precede a j'
  a[i]      : continua >= 0  — tiempo inicio descarga camión entrada i
  d[j]      : continua >= 0  — tiempo inicio carga camión salida j
  Cmax      : continua >= 0  — makespan

Restricciones (10 grupos, ~13 líneas en AMPL):
  C1  Cmax >= d[j] + LJ[j]                          para todo j
  C2  sum_j x[i,j,k] = ri[i,k]                      para todo i,k
  C3  sum_i x[i,j,k] = sj[j,k]                      para todo j,k
  C4  sum_k x[i,j,k] <= M_ij * v[i,j]               para todo i,j
  C5  alpha[i,i'] + alpha[i',i] = 1                  para i < i'
  C6  No reflexividad (dominio i≠i')
  C7  a[i'] >= a[i] + DI[i] + t_sw - M*(1-alpha[i,i'])  para i≠i'
  C8  beta[j,j'] + beta[j',j] = 1                    para j < j'
  C9  No reflexividad salida
  C10 d[j'] >= d[j] + LJ[j] + t_sw - M*(1-beta[j,j'])   para j≠j'
  C11 d[j]  >= a[i] + DI[i] + t_tr - M*(1-v[i,j])       para todo i,j
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import time

# ─── Parámetros operativos ────────────────────────────────────────────────────
T_UNIT   = 1       # min por unidad (carga/descarga)
T_TRANS  = 5       # min traslado interno por lote
T_SWITCH = 10      # min cambio de camión en muelle
BIG_M    = 100_000


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CrossDockInstance:
    """Instancia del problema de Cross Docking."""
    num_inbound:  int = 0
    num_outbound: int = 0
    num_products: int = 0
    ri: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (camión_entrada, producto) → qty
    sj: Dict[Tuple[int, int], int] = field(default_factory=dict)  # (camión_salida,  producto) → qty

    def inbound_trucks(self)  -> List[int]: return list(range(1, self.num_inbound  + 1))
    def outbound_trucks(self) -> List[int]: return list(range(1, self.num_outbound + 1))
    def products(self)        -> List[int]: return list(range(1, self.num_products + 1))

    def validate_balance(self) -> List[str]:
        """Devuelve lista de errores de balance oferta-demanda por producto."""
        errors = []
        for k in self.products():
            supply = sum(self.ri.get((i, k), 0) for i in self.inbound_trucks())
            demand = sum(self.sj.get((j, k), 0) for j in self.outbound_trucks())
            if supply != demand:
                errors.append(f"Producto {k}: oferta={supply} ≠ demanda={demand}")
        return errors


@dataclass
class SolverResult:
    status:         str
    makespan:       float
    inbound_order:  List[int]                    # camiones en orden de atención
    outbound_order: List[int]
    a:  Dict[int, float]                         # tiempo inicio descarga por camión entrada
    d:  Dict[int, float]                         # tiempo inicio carga por camión salida
    x:  Dict[Tuple[int, int, int], int]          # flujo x[i,j,k]
    v:  Dict[Tuple[int, int], int]               # transferencias directas v[i,j]
    di: Dict[int, float]                         # duración descarga camión i
    lj: Dict[int, float]                         # duración carga camión j
    solver_used: str   = ""
    solve_time:  float = 0.0
    gap:         float = 0.0
    message:     str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 2 — PARSER FORMATO TS5
# ══════════════════════════════════════════════════════════════════════════════

def parse_ts5(content: str) -> CrossDockInstance:
    """
    Parsea texto con formato TS5 (separado por espacios o tabulaciones).

    Formato:
        i  <num_entrada>
        o  <num_salida>
        n  <num_productos>
        r  <camion>  <producto>  <cantidad>   ← camión de entrada
        s  <camion>  <producto>  <cantidad>   ← camión de salida
    """
    inst   = CrossDockInstance()
    tokens = content.split()
    idx, n = 0, len(tokens)

    while idx < n:
        tok = tokens[idx]
        if tok == 'i' and idx + 1 < n:
            inst.num_inbound  = int(tokens[idx + 1]); idx += 2
        elif tok == 'o' and idx + 1 < n:
            inst.num_outbound = int(tokens[idx + 1]); idx += 2
        elif tok == 'n' and idx + 1 < n:
            inst.num_products = int(tokens[idx + 1]); idx += 2
        elif tok == 'r' and idx + 3 < n:
            truck, prod, qty = int(tokens[idx+1]), int(tokens[idx+2]), int(tokens[idx+3])
            inst.ri[(truck, prod)] = inst.ri.get((truck, prod), 0) + qty
            idx += 4
        elif tok == 's' and idx + 3 < n:
            truck, prod, qty = int(tokens[idx+1]), int(tokens[idx+2]), int(tokens[idx+3])
            inst.sj[(truck, prod)] = inst.sj.get((truck, prod), 0) + qty
            idx += 4
        else:
            idx += 1

    return inst


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — SOLVER CON amplpy (AMPL + HiGHS/CPLEX/Gurobi)
# ══════════════════════════════════════════════════════════════════════════════

_AMPL_MODEL = """
set I;
set J;
set K;

param ri{I, K} default 0;
param sj{J, K} default 0;

param t_unit   := 1;
param t_trans  := 5;
param t_switch := 10;
param BIG_M    := 100000;

param DI{i in I} = sum{k in K} ri[i,k] * t_unit;
param LJ{j in J} = sum{k in K} sj[j,k] * t_unit;

var x{I, J, K} >= 0, integer;
var v{I, J}    binary;
var alpha{i in I, i2 in I : i <> i2} binary;
var beta {j in J, j2 in J : j <> j2} binary;
var a{I} >= 0;
var d{J} >= 0;
var Cmax >= 0;

minimize makespan: Cmax;

subject to c1 {j in J}:
    Cmax >= d[j] + LJ[j];

subject to c2 {i in I, k in K}:
    sum{j in J} x[i,j,k] = ri[i,k];

subject to c3 {j in J, k in K}:
    sum{i in I} x[i,j,k] = sj[j,k];

subject to c4 {i in I, j in J}:
    sum{k in K} x[i,j,k] <= (sum{k in K} (ri[i,k] + sj[j,k])) * v[i,j];

subject to c5 {i in I, i2 in I : i < i2}:
    alpha[i,i2] + alpha[i2,i] = 1;

subject to c7 {i in I, i2 in I : i <> i2}:
    a[i2] >= a[i] + DI[i] + t_switch - BIG_M * (1 - alpha[i,i2]);

subject to c8 {j in J, j2 in J : j < j2}:
    beta[j,j2] + beta[j2,j] = 1;

subject to c9 {j in J, j2 in J : j <> j2}:
    d[j2] >= d[j] + LJ[j] + t_switch - BIG_M * (1 - beta[j,j2]);

subject to c11 {i in I, j in J}:
    d[j] >= a[i] + DI[i] + t_trans - BIG_M * (1 - v[i,j]);
"""


def _solve_amplpy(inst: CrossDockInstance, solver: str,
                  time_limit: int, mip_gap: float) -> Optional[SolverResult]:
    try:
        from amplpy import AMPL
    except ImportError:
        return None

    try:
        import pandas as pd

        ampl = AMPL()
        ampl.eval(_AMPL_MODEL)

        ampl.set["I"] = inst.inbound_trucks()
        ampl.set["J"] = inst.outbound_trucks()
        ampl.set["K"] = inst.products()

        rows_r = [(i, k, v) for (i, k), v in inst.ri.items() if v > 0]
        rows_s = [(j, k, v) for (j, k), v in inst.sj.items() if v > 0]
        if rows_r:
            df = pd.DataFrame(rows_r, columns=["i","k","val"]).set_index(["i","k"])
            ampl.param["ri"] = df["val"]
        if rows_s:
            df = pd.DataFrame(rows_s, columns=["j","k","val"]).set_index(["j","k"])
            ampl.param["sj"] = df["val"]

        ampl.option["solver"] = solver
        opts = {
            "highs":  f"mip_rel_gap={mip_gap} time_limit={time_limit}",
            "cplex":  f"mipgap={mip_gap} timelimit={time_limit}",
            "gurobi": f"mipgap={mip_gap} timelimit={time_limit}",
            "cbc":    f"ratioGap={mip_gap} sec={time_limit}",
        }
        if solver in opts:
            ampl.option[f"{solver}_options"] = opts[solver]

        t0 = time.time()
        ampl.solve()
        elapsed = time.time() - t0

        sr = ampl.option["solve_result"]
        if sr not in ("solved", "solved?", "limit"):
            return SolverResult(
                status=f"infeasible ({sr})", makespan=0,
                inbound_order=[], outbound_order=[],
                a={}, d={}, x={}, v={}, di={}, lj={},
                solver_used=f"amplpy/{solver}", message=sr
            )

        I = inst.inbound_trucks()
        J = inst.outbound_trucks()
        K = inst.products()

        a_v  = {i: ampl.var["a"][i].value() or 0.0  for i in I}
        d_v  = {j: ampl.var["d"][j].value() or 0.0  for j in J}
        cmax = ampl.var["Cmax"].value() or 0.0

        di = {i: sum(inst.ri.get((i,k),0) for k in K) * T_UNIT for i in I}
        lj = {j: sum(inst.sj.get((j,k),0) for k in K) * T_UNIT for j in J}

        x_v = {}
        for i in I:
            for j in J:
                for k in K:
                    val = ampl.var["x"][i,j,k].value()
                    if val and val > 0.5:
                        x_v[(i,j,k)] = round(val)

        v_v = {}
        for i in I:
            for j in J:
                val = ampl.var["v"][i,j].value()
                if val and val > 0.5:
                    v_v[(i,j)] = 1

        return SolverResult(
            status="optimal", makespan=cmax,
            inbound_order=sorted(I, key=lambda i: a_v[i]),
            outbound_order=sorted(J, key=lambda j: d_v[j]),
            a=a_v, d=d_v, x=x_v, v=v_v, di=di, lj=lj,
            solver_used=f"amplpy/{solver}", solve_time=elapsed,
        )

    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — FALLBACK: PuLP / CBC (open-source, sin licencia)
# ══════════════════════════════════════════════════════════════════════════════

def _solve_pulp(inst: CrossDockInstance,
                time_limit: int, mip_gap: float) -> SolverResult:
    import pulp

    I = inst.inbound_trucks()
    J = inst.outbound_trucks()
    K = inst.products()

    DI = {i: sum(inst.ri.get((i,k), 0) for k in K) * T_UNIT for i in I}
    LJ = {j: sum(inst.sj.get((j,k), 0) for k in K) * T_UNIT for j in J}

    prob = pulp.LpProblem("CrossDocking", pulp.LpMinimize)

    Cmax  = pulp.LpVariable("Cmax", lowBound=0)
    x     = {(i,j,k): pulp.LpVariable(f"x_{i}_{j}_{k}", lowBound=0, cat="Integer")
             for i in I for j in J for k in K}
    v     = {(i,j): pulp.LpVariable(f"v_{i}_{j}", cat="Binary")
             for i in I for j in J}
    alpha = {(i,i2): pulp.LpVariable(f"al_{i}_{i2}", cat="Binary")
             for i in I for i2 in I if i != i2}
    beta  = {(j,j2): pulp.LpVariable(f"be_{j}_{j2}", cat="Binary")
             for j in J for j2 in J if j != j2}
    a     = {i: pulp.LpVariable(f"a_{i}", lowBound=0) for i in I}
    d     = {j: pulp.LpVariable(f"d_{j}", lowBound=0) for j in J}

    # Objetivo
    prob += Cmax

    # C1
    for j in J:
        prob += Cmax >= d[j] + LJ[j]

    # C2 — conservación oferta
    for i in I:
        for k in K:
            prob += pulp.lpSum(x[i,j,k] for j in J) == inst.ri.get((i,k), 0)

    # C3 — satisfacción demanda
    for j in J:
        for k in K:
            prob += pulp.lpSum(x[i,j,k] for i in I) == inst.sj.get((j,k), 0)

    # C4 — vinculación x → v
    for i in I:
        for j in J:
            cap = sum(inst.ri.get((i,k),0) + inst.sj.get((j,k),0) for k in K)
            if cap > 0:
                prob += pulp.lpSum(x[i,j,k] for k in K) <= cap * v[i,j]

    # C5 — par de secuencia entrada
    for i in I:
        for i2 in I:
            if i < i2:
                prob += alpha[i,i2] + alpha[i2,i] == 1

    # C7 — tiempo inicio entrada con Big-M
    for i in I:
        for i2 in I:
            if i != i2:
                prob += a[i2] >= a[i] + DI[i] + T_SWITCH - BIG_M * (1 - alpha[i,i2])

    # C8 — par de secuencia salida
    for j in J:
        for j2 in J:
            if j < j2:
                prob += beta[j,j2] + beta[j2,j] == 1

    # C9 — tiempo inicio salida con Big-M
    for j in J:
        for j2 in J:
            if j != j2:
                prob += d[j2] >= d[j] + LJ[j] + T_SWITCH - BIG_M * (1 - beta[j,j2])

    # C11 — sincronización entrada→salida
    for i in I:
        for j in J:
            prob += d[j] >= a[i] + DI[i] + T_TRANS - BIG_M * (1 - v[i,j])

    # Resolver
    t0 = time.time()
    prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, gapRel=mip_gap, msg=0))
    elapsed = time.time() - t0

    obj = pulp.value(prob.objective)
    if obj is None:
        return SolverResult(
            status="infeasible", makespan=0,
            inbound_order=[], outbound_order=[],
            a={}, d={}, x={}, v={}, di=DI, lj=LJ,
            solver_used="PuLP/CBC", solve_time=elapsed,
            message="No se encontró solución factible"
        )

    a_v    = {i: max(0.0, pulp.value(a[i]) or 0.0)  for i in I}
    d_v    = {j: max(0.0, pulp.value(d[j]) or 0.0)  for j in J}
    cmax_v = pulp.value(Cmax) or 0.0

    x_v = {}
    for i in I:
        for j in J:
            for k in K:
                val = pulp.value(x[i,j,k])
                if val and val > 0.5:
                    x_v[(i,j,k)] = round(val)

    v_v = {}
    for i in I:
        for j in J:
            val = pulp.value(v[i,j])
            if val and val > 0.5:
                v_v[(i,j)] = 1

    return SolverResult(
        status="optimal", makespan=cmax_v,
        inbound_order=sorted(I, key=lambda i: a_v[i]),
        outbound_order=sorted(J, key=lambda j: d_v[j]),
        a=a_v, d=d_v, x=x_v, v=v_v, di=DI, lj=LJ,
        solver_used="PuLP/CBC", solve_time=elapsed,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 5 — FUNCIÓN PÚBLICA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def solve(inst: CrossDockInstance,
          preferred_solver: str = "highs",
          time_limit: int = 300,
          mip_gap: float = 0.001) -> SolverResult:
    """
    Resuelve la instancia de Cross Docking MIP.
    Intenta amplpy con el solver indicado; si no está disponible usa PuLP/CBC.

    Args:
        inst:             Instancia parseada con parse_ts5()
        preferred_solver: 'highs' | 'cplex' | 'gurobi' | 'cbc'
        time_limit:       Segundos máximos de cómputo
        mip_gap:          Gap relativo MIP aceptable

    Returns:
        SolverResult con makespan, órdenes, flujos y tiempos
    """
    result = _solve_amplpy(inst, solver=preferred_solver,
                           time_limit=time_limit, mip_gap=mip_gap)
    if result is not None:
        return result
    return _solve_pulp(inst, time_limit=time_limit, mip_gap=mip_gap)
