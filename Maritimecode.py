from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

from pulp import (
    LpProblem, LpVariable, LpMinimize, lpSum,
    LpBinary, PULP_CBC_CMD, LpStatus
)

# -----------------------------
# Paths (relative to THIS script)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

VESSEL_MOVEMENTS_CSV = BASE_DIR / "vessel_movements_dataset.csv"
LLAF_TABLE_CSV       = BASE_DIR / "llaf_table.csv"
FACTORS_XLSX         = BASE_DIR / "calculation_factors.xlsx"


def require_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            f"Fix: place it in {BASE_DIR} or change the path in the script."
        )


def norm_fuel(s: str) -> str:
    return str(s).strip().upper()


def dwt_bucket(dwt: float) -> str:
    if dwt <= 40000:
        return "10-40k DWT"
    if dwt <= 55000:
        return "40-55k DWT"
    if dwt <= 80000:
        return "55-80k DWT"
    if dwt <= 120000:
        return "80-120k DWT"
    return ">120 DWT"


@dataclass
class FactorTables:
    lcv_mj_per_kg: Dict[str, float]
    cf_co2: Dict[str, float]
    cf_ch4: Dict[str, float]
    cf_n2o: Dict[str, float]
    usd_per_gj: Dict[str, float]
    carbon_usd_per_t_co2eq: float
    base_capex_musd_by_bucket: Dict[str, float]
    capex_multiplier_by_fuel: Dict[str, float]
    safety_rate: Dict[int, float]
    llaf: Dict[int, Dict[str, float]]


def load_factor_tables(factors_xlsx: Path, llaf_csv: Path) -> FactorTables:
    xl = pd.ExcelFile(factors_xlsx)

    cf = xl.parse("Cf").dropna(subset=["Fuel Type"]).copy()
    cf["fuel_norm"] = cf["Fuel Type"].map(norm_fuel)

    lcv = dict(zip(cf["fuel_norm"], cf["LCV (MJ/kg)"].astype(float)))
    cf_co2 = dict(zip(cf["fuel_norm"], cf["Cf_CO2"].astype(float)))
    cf_ch4 = dict(zip(cf["fuel_norm"], cf["Cf_CH4"].astype(float)))
    cf_n2o = dict(zip(cf["fuel_norm"], cf["Cf_N2O"].astype(float)))

    fc = xl.parse("Fuel cost").dropna(subset=["Fuel Type"]).copy()
    fc["fuel_norm"] = fc["Fuel Type"].map(norm_fuel)
    usd_per_gj = dict(zip(fc["fuel_norm"], fc["Cost per GJ (USD)"].astype(float)))

    carbon_usd_per_t = float(xl.parse("Cost of Carbon").iloc[0, 1])

    ship = xl.parse("Cost of ship")
    bucket_headers = ship.iloc[0, 1:6].tolist()
    distillate_costs = ship.iloc[2, 1:6].astype(float).tolist()
    base_capex_musd_by_bucket = dict(zip(bucket_headers, distillate_costs))

    mult = ship.iloc[4:11, 0:6].copy()
    mult.columns = ["Fuel Type", "b1", "b2", "b3", "b4", "b5"]
    mult["fuel_norm"] = mult["Fuel Type"].map(norm_fuel)
    capex_multiplier_by_fuel = dict(zip(mult["fuel_norm"], mult["b1"].astype(float)))
    capex_multiplier_by_fuel["DISTILLATE FUEL"] = 1.0

    ssa = xl.parse("Safety score adjustment").dropna(subset=["Safety score"]).iloc[:5].copy()
    ssa["score"] = ssa["Safety score"].astype(int)
    ssa["rate"] = pd.to_numeric(ssa["Adjustment rate (%)"], errors="coerce").astype(float) / 100.0
    safety_rate = dict(zip(ssa["score"], ssa["rate"]))

    llaf_df = pd.read_csv(llaf_csv)
    llaf_map: Dict[int, Dict[str, float]] = {}
    for _, r in llaf_df.iterrows():
        pct = int(str(r["Load"]).replace("%", ""))
        llaf_map[pct] = {"CO2": float(r["CO2"]), "CH4": float(r["CH4"]), "N2O": float(r["N2O"])}

    return FactorTables(
        lcv_mj_per_kg=lcv,
        cf_co2=cf_co2,
        cf_ch4=cf_ch4,
        cf_n2o=cf_n2o,
        usd_per_gj=usd_per_gj,
        carbon_usd_per_t_co2eq=carbon_usd_per_t,
        base_capex_musd_by_bucket=base_capex_musd_by_bucket,
        capex_multiplier_by_fuel=capex_multiplier_by_fuel,
        safety_rate=safety_rate,
        llaf=llaf_map,
    )


def compute_ship_capex_musd(dwt: float, main_fuel_norm: str, ft: FactorTables) -> float:
    base = float(ft.base_capex_musd_by_bucket[dwt_bucket(dwt)])
    mult = float(ft.capex_multiplier_by_fuel.get(main_fuel_norm, 1.0))
    return base * mult


def llaf_factor(ft: FactorTables, pct_lf: int, gas: str) -> float:
    if pct_lf > 20:
        return 1.0
    return float(ft.llaf.get(int(pct_lf), {"CO2": 1.0, "CH4": 1.0, "N2O": 1.0})[gas])


def compute_vessel_summaries(movements_csv: Path, ft: FactorTables) -> pd.DataFrame:
    """
    One row per vessel_id for the Singapore -> Port Hedland segment.
    IMPORTANT: Fuel cost follows the clarification:
      cost_per_tonne = USD_per_GJ * LCV (by each machinery fuel type),
      total fuel cost = sum over machinery (fuel_tonnes * cost_per_tonne).
    """
    df = pd.read_csv(movements_csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    df["speed_knots"] = pd.to_numeric(df["speed_knots"], errors="coerce")
    df["in_port"] = df["in_port_boundary"].notna()
    df["in_anch"] = df["in_anchorage"].notna()

    df["mode"] = np.select(
        [
            df["in_anch"] & (df["speed_knots"] < 1),
            df["in_port"] & (df["speed_knots"] > 1),
            (~df["in_port"]) & (df["speed_knots"] >= 1),
        ],
        ["anchorage", "maneuver", "transit"],
        default="drifting",
    )

    ms = df.loc[df["speed_knots"] >= 1].groupby("vessel_id")["speed_knots"].max()

    for col in ["main_engine_fuel_type", "aux_engine_fuel_type", "boil_engine_fuel_type"]:
        df[col + "_norm"] = df[col].map(norm_fuel)

    LCV_DIST = float(ft.lcv_mj_per_kg.get("DISTILLATE FUEL", 42.7))

    def get_lcv(fuel_norm: str) -> float:
        return float(ft.lcv_mj_per_kg.get(fuel_norm, LCV_DIST))

    def cf_lookup(mapd: Dict[str, float], fuel_norm: str) -> float:
        return float(mapd.get(fuel_norm, mapd["DISTILLATE FUEL"]))

    # Clarified Step 6a:
    # cost per tonne = cost per GJ * LCV (LCV in MJ/kg equals GJ/tonne numerically)
    def cost_per_tonne(fuel_norm: str) -> float:
        return float(ft.usd_per_gj.get(fuel_norm, ft.usd_per_gj["DISTILLATE FUEL"])) * get_lcv(fuel_norm)

    # Ownership constants
    r = 0.08
    N = 30
    CRF = (r * (1 + r) ** N) / (((1 + r) ** N) - 1)

    vessel_rows: List[dict] = []

    for vid, g in df.groupby("vessel_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)

        sg_idx = g.index[g["in_port_boundary"].eq("Singapore")].to_list()
        ph_idx = g.index[g["in_port_boundary"].eq("Port Hedland")].to_list()
        if not sg_idx or not ph_idx:
            continue

        candidate_ph = [i for i in ph_idx if i > min(sg_idx)]
        if not candidate_ph:
            continue
        end = min(candidate_ph)
        start = max([i for i in sg_idx if i < end])

        voyage = g.loc[start:end].copy()

        voyage["next_ts"] = voyage["timestamp"].shift(-1)
        voyage["A_hours"] = (voyage["next_ts"] - voyage["timestamp"]).dt.total_seconds() / 3600.0
        voyage = voyage.dropna(subset=["A_hours"])
        if voyage.empty:
            continue

        scope_mask = voyage["mode"].isin(["transit", "maneuver"])
        voyage["A_scope"] = voyage["A_hours"].where(scope_mask, 0.0)

        MS = float(ms.get(vid, float(voyage["speed_knots"].max())))
        voyage["LF"] = ((voyage["speed_knots"] / MS) ** 3).round(2)
        voyage.loc[(voyage["LF"] < 0.02) & scope_mask, "LF"] = 0.02
        voyage.loc[~scope_mask, "LF"] = 0.0

        pctLF = (voyage["LF"] * 100).round().astype(int)
        pctLF = pctLF.where(~((pctLF < 2) & scope_mask), 2)

        for gas in ["CO2", "CH4", "N2O"]:
            voyage[f"LLAF_{gas}"] = [
                1.0 if (not sm) else llaf_factor(ft, int(p), gas)
                for p, sm in zip(pctLF, scope_mask)
            ]

        voyage["sfc_me_adj"] = voyage["sfc_me"] * (LCV_DIST / voyage["main_engine_fuel_type_norm"].map(get_lcv))
        voyage["sfc_ae_adj"] = voyage["sfc_ae"] * (LCV_DIST / voyage["aux_engine_fuel_type_norm"].map(get_lcv))
        voyage["sfc_ab_adj"] = voyage["sfc_ab"] * (LCV_DIST / voyage["boil_engine_fuel_type_norm"].map(get_lcv))

        voyage["fc_me"] = (voyage["mep"] * voyage["LF"] * voyage["A_scope"] * voyage["sfc_me_adj"]) / 1e6
        voyage["fc_ae"] = (voyage["ael"] * voyage["A_scope"] * voyage["sfc_ae_adj"]) / 1e6
        voyage["fc_ab"] = (voyage["abl"] * voyage["A_scope"] * voyage["sfc_ab_adj"]) / 1e6

        mf = voyage["main_engine_fuel_type_norm"].iloc[0]
        af = voyage["aux_engine_fuel_type_norm"].iloc[0]
        bf = voyage["boil_engine_fuel_type_norm"].iloc[0]

        # Emissions
        for gas, cf_map in [("CO2", ft.cf_co2), ("CH4", ft.cf_ch4), ("N2O", ft.cf_n2o)]:
            voyage[f"em_{gas}_me"] = voyage["fc_me"] * cf_lookup(cf_map, mf) * voyage[f"LLAF_{gas}"]
            voyage[f"em_{gas}_ae"] = voyage["fc_ae"] * cf_lookup(cf_map, af) * voyage[f"LLAF_{gas}"]
            voyage[f"em_{gas}_ab"] = voyage["fc_ab"] * cf_lookup(cf_map, bf) * voyage[f"LLAF_{gas}"]

        fc_me = float(voyage["fc_me"].sum())
        fc_ae = float(voyage["fc_ae"].sum())
        fc_ab = float(voyage["fc_ab"].sum())
        total_fc = fc_me + fc_ae + fc_ab

        total_co2 = float(voyage[["em_CO2_me", "em_CO2_ae", "em_CO2_ab"]].sum().sum())
        total_ch4 = float(voyage[["em_CH4_me", "em_CH4_ae", "em_CH4_ab"]].sum().sum())
        total_n2o = float(voyage[["em_N2O_me", "em_N2O_ae", "em_N2O_ab"]].sum().sum())
        co2eq = total_co2 + 28.0 * total_ch4 + 265.0 * total_n2o

        # Clarified Step 6a: per-machinery fuel cost
        cpt_me = cost_per_tonne(mf)
        cpt_ae = cost_per_tonne(af)
        cpt_ab = cost_per_tonne(bf)

        fuel_cost = (fc_me * cpt_me) + (fc_ae * cpt_ae) + (fc_ab * cpt_ab)

        carbon_cost = co2eq * ft.carbon_usd_per_t_co2eq

        dwt_val = float(voyage["dwt"].iloc[0])
        safety = int(voyage["safety_score"].iloc[0])

        # monthly ownership
        P = compute_ship_capex_musd(dwt_val, mf, ft)  # million USD
        S = 0.10 * P
        annual_own_musd = ((P - S) * CRF) + (r * S)
        monthly_own_usd = annual_own_musd * 1e6 / 12.0

        base_monthly_cost = fuel_cost + carbon_cost + monthly_own_usd
        risk_adj = float(ft.safety_rate.get(safety, 0.0))
        risk_adjusted_cost = base_monthly_cost * (1.0 + risk_adj)

        vessel_rows.append(
            dict(
                vessel_id=vid,
                dwt=dwt_val,
                safety_score=safety,
                main_engine_fuel_type=mf,

                # fuel breakdown (audit)
                fuel_me_t=fc_me,
                fuel_ae_t=fc_ae,
                fuel_ab_t=fc_ab,
                fuel_total_t=total_fc,

                cost_per_tonne_me=cpt_me,
                cost_per_tonne_ae=cpt_ae,
                cost_per_tonne_ab=cpt_ab,

                fuel_cost_usd=fuel_cost,
                carbon_cost_usd=carbon_cost,
                ownership_cost_usd=monthly_own_usd,
                base_cost_usd=base_monthly_cost,
                risk_adj_rate=risk_adj,

                co2eq_t=co2eq,
                cost_usd=risk_adjusted_cost,
            )
        )

    out = pd.DataFrame(vessel_rows)
    if out.empty:
        return out

    out["cost_per_dwt"] = out["cost_usd"] / out["dwt"]
    out["rank_cost_per_dwt_within_fuel"] = out.groupby("main_engine_fuel_type")["cost_per_dwt"].rank(method="min")
    return out


def solve_fleet_pulp(vessel_summary: pd.DataFrame, demand_dwt_tonnes: float, min_avg_safety: float) -> dict:
    df = vessel_summary.reset_index(drop=True).copy()
    fuels = sorted(df["main_engine_fuel_type"].unique())
    n = len(df)

    prob = LpProblem("SmartFleetSelection", LpMinimize)
    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in range(n)]

    prob += lpSum(df.loc[i, "cost_usd"] * x[i] for i in range(n))  # objective

    prob += lpSum(df.loc[i, "dwt"] * x[i] for i in range(n)) >= demand_dwt_tonnes  # demand

    prob += (
        lpSum(df.loc[i, "safety_score"] * x[i] for i in range(n))
        - float(min_avg_safety) * lpSum(x[i] for i in range(n))
        >= 0.0
    )  # avg safety

    for f in fuels:
        idx = df.index[df["main_engine_fuel_type"] == f].tolist()
        prob += lpSum(x[i] for i in idx) >= 1  # 1 per fuel type

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if LpStatus[status] != "Optimal":
        raise RuntimeError(f"MILP not optimal. Status = {LpStatus[status]}")

    chosen = [i for i in range(n) if x[i].value() is not None and x[i].value() > 0.5]
    sel = df.loc[chosen].copy()

    return dict(
        selected=sel,
        total_dwt=float(sel["dwt"].sum()),
        total_cost_usd=float(sel["cost_usd"].sum()),
        avg_safety=float(sel["safety_score"].mean()),
        num_unique_main_fuel_types=int(sel["main_engine_fuel_type"].nunique()),
        fleet_size=int(len(sel)),
        total_co2eq_t=float(sel["co2eq_t"].sum()),
        total_fuel_consumption_t=float(sel["fuel_total_t"].sum()),
    )


def print_outputs(label: str, out: dict) -> None:
    print(f"\n=== {label} ===")
    print(f"Total DWT: {out['total_dwt']:.0f}")
    print(f"Fleet size: {out['fleet_size']}")
    print(f"Avg safety: {out['avg_safety']:.2f}")
    print(f"Unique fuels: {out['num_unique_main_fuel_types']}")
    print(f"Total cost (USD): {out['total_cost_usd']:.2f}")
    print(f"Total CO2e (t): {out['total_co2eq_t']:.3f}")
    print(f"Total fuel (t): {out['total_fuel_consumption_t']:.3f}")


if __name__ == "__main__":
    require_file(VESSEL_MOVEMENTS_CSV)
    require_file(LLAF_TABLE_CSV)
    require_file(FACTORS_XLSX)

    ft = load_factor_tables(FACTORS_XLSX, LLAF_TABLE_CSV)

    vessel_summary = compute_vessel_summaries(VESSEL_MOVEMENTS_CSV, ft)
    if vessel_summary.empty:
        raise RuntimeError("No valid Singapore -> Port Hedland voyages found.")

    # Save candidate pool with audit columns
    cand_csv = BASE_DIR / "all_candidate_vessels_with_costs.csv"
    vessel_summary.sort_values(["main_engine_fuel_type", "cost_per_dwt"]).to_csv(cand_csv, index=False)
    print(f"Saved: {cand_csv}  (n={vessel_summary['vessel_id'].nunique()})")

    # Monthly bunker demand (tonnes)
    D = 54.92e6 / 12.0

    # Baseline and sensitivity
    out3 = solve_fleet_pulp(vessel_summary, demand_dwt_tonnes=D, min_avg_safety=3.0)
    print_outputs("Baseline (avg safety >= 3)", out3)
    out3["selected"].sort_values(["main_engine_fuel_type", "cost_per_dwt"]).to_csv(
        BASE_DIR / "selected_fleet_baseline_safety3.csv", index=False
    )

    out4 = solve_fleet_pulp(vessel_summary, demand_dwt_tonnes=D, min_avg_safety=4.0)
    print_outputs("Sensitivity (avg safety >= 4)", out4)
    out4["selected"].sort_values(["main_engine_fuel_type", "cost_per_dwt"]).to_csv(
        BASE_DIR / "selected_fleet_safety4.csv", index=False
    )

    print("\nSaved fleet CSVs:")
    print(" - selected_fleet_baseline_safety3.csv")
    print(" - selected_fleet_safety4.csv")
