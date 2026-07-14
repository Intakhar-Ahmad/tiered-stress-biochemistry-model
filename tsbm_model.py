"""
Tiered Stress Biochemistry Model (TSBM)
======================================
Ten coupled ordinary differential equations. Time unit: hours. Default horizon: 168 h.

State vector:
    [NE, VitC, Ald, Mg, BDNF, Nrf2, INF, Trp, Kyn, Cstress]

Cstress is a slow cortisol-elevation state used as a one-dimensional proxy for
endocrine allostatic burden. It is not intended to represent a full
multidimensional allostatic-load score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional
import numpy as np
from scipy.integrate import solve_ivp

STATES = ["NE", "VitC", "Ald", "Mg", "BDNF", "Nrf2", "INF", "Trp", "Kyn", "Cstress"]
IDX = {name: i for i, name in enumerate(STATES)}

# Baseline state values.
NE0 = 1.0                 # relative units
VitC0 = 60.0              # micromolar, serum/plasma-comparable reference
Ald0 = 8.0                # arbitrary units, phenomenological stress/RAAS drive
Mg0 = 0.85                # mmol/L, serum-comparable exchangeable-pool proxy
TRP0 = 60.0               # micromolar
BDNF0 = 100.0             # percent of normal
NRF0 = 100.0              # percent of normal

THRESHOLDS = {
    "Mg_hypo": 0.65,
    "VitC_def": 23.0,
    "BDNF_70": 70.0,
    "BDNF_60": 60.0,
    "KYNTRP_boundary": 0.08,
}

DEFAULT_PARAMETERS: Dict[str, float] = {
    "t_peak": 8.0,
    # Eq 1: locus coeruleus / noradrenergic drive
    "k_ne_on": 2.0,
    "k_ne_off": 4.0,
    "k_thind": 0.020,
    # Eq 2: vitamin C
    "kin_vitc": 0.023,
    "kc_vitc": 0.105,
    "kc_ne": 0.190,
    "Cb_vitc": 8.0,
    # Eq 3: aldosterone
    "ka_ald": 0.55,
    "kcl_ald": 0.20,
    "th_ald": 18.5,
    "KMg_ald": 0.6,
    # Eq 4: magnesium
    "kin_mg": 0.016,
    "kw_mg": 0.040,
    "Mg_floor": 0.55,
    # Eq 5: BDNF
    "krec_bdnf": 0.010,
    "ksil_bdnf": 0.100,
    "K_bdnf": 15.0,
    # Eq 6: Nrf2
    "krec_nrf": 0.010,
    "ksil_nrf": 0.130,
    "K_nrf": 14.0,
    # Eq 7: inflammation
    "kinf": 0.0060,
    "kauto": 0.130,
    "kcl_inf": 0.012,
    "th_inf": 11.0,
    # Eq 8 and Eq 9: tryptophan / kynurenine
    "kin_trp": 0.010,
    "kido": 0.0007,
    "kcl_kyn": 0.080,
    "kyn_base": 0.168,
    # Eq 10: slow cortisol-elevation state
    "kacc": 0.028,
    "kdec": 0.0035,
    "th_acc": 12.0,
    "Cmax": 18.0,
}

PHENOTYPES = {
    "normal":    {"C0": 10.0, "C1": 7.5, "acc": 0.0, "Cstr0": 0.0, "acute": 0.0,  "psi": "none", "extra": 0.0},
    "acute":     {"C0": 10.0, "C1": 7.5, "acc": 0.0, "Cstr0": 0.0, "acute": 20.0, "psi": "none", "extra": 0.0},
    "chronic":   {"C0": 10.0, "C1": 7.5, "acc": 1.0, "Cstr0": 6.0, "acute": 0.0,  "psi": "none", "extra": 0.0},
    "depressed": {"C0": 13.0, "C1": 8.0, "acc": 1.0, "Cstr0": 3.0, "acute": 0.0,  "psi": "dep",  "extra": 0.0},
    "ptsd":      {"C0": 7.5,  "C1": 5.5, "acc": 0.0, "Cstr0": 0.0, "acute": 0.0,  "psi": "ptsd", "extra": 0.0},
}

PHENOTYPE_ORDER = ["normal", "acute", "chronic", "depressed", "ptsd"]


def parameters(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Return a copy of the default parameter dictionary with optional overrides."""
    p = dict(DEFAULT_PARAMETERS)
    if overrides:
        p.update(overrides)
    return p


def phenotype(name: str, overrides: Optional[Dict[str, float | str]] = None) -> Dict[str, float | str]:
    """Return a phenotype dictionary with optional overrides."""
    if name not in PHENOTYPES:
        raise ValueError(f"Unknown phenotype {name!r}. Choose from {list(PHENOTYPES)}")
    ph = dict(PHENOTYPES[name])
    if overrides:
        ph.update(overrides)
    return ph


def c_circadian(t: float, ph: Dict[str, float | str], p: Dict[str, float]) -> float:
    return float(ph["C0"] + ph["C1"] * np.cos(2 * np.pi * (t - p["t_peak"]) / 24.0))


def c_acute(t: float, ph: Dict[str, float | str]) -> float:
    amp = float(ph.get("acute", 0.0))
    if amp <= 0 or t < 8.0:
        return 0.0
    return amp * np.exp(-(t - 8.0) / 0.5)


def psi_multiplier(kind: str, bdnf: float, dep_gain: float = 0.60, ptsd_gain: float = 0.15) -> float:
    deficit = max(0.0, (100.0 - bdnf) / 100.0)
    if kind == "dep":
        return 1.0 + dep_gain * deficit
    if kind == "ptsd":
        return 1.0 - ptsd_gain * deficit
    return 1.0


def rhs(t: float, y: np.ndarray, ph: Dict[str, float | str], p: Dict[str, float], dep_gain: float = 0.60, ptsd_gain: float = 0.15) -> np.ndarray:
    """Right-hand side of the ten-equation TSBM system.

    This implementation is intentionally index-based rather than dictionary-based
    because robustness analyses evaluate the ODE many times.
    """
    NE = y[0]; VitC = y[1]; Ald = y[2]; Mg = y[3]; BDNF = y[4]
    Nrf2 = y[5]; INF = y[6]; Trp = y[7]; Kyn = y[8]; Cstress = y[9]
    C0 = float(ph["C0"])
    Cacute = c_acute(t, ph)
    Ctot = c_circadian(t, ph, p) + Cstress + Cacute + float(ph.get("extra", 0.0))
    dy = np.zeros(10, dtype=float)

    # Eq 1: locus coeruleus noradrenergic drive
    th_act = 1.0 + p["k_thind"] * Cstress
    drive_lc = (Cacute + max(Ctot - C0, 0.0)) / 10.0
    dy[0] = p["k_ne_on"] * th_act * max(drive_lc, 0.0) - p["k_ne_off"] * (NE - NE0)

    # Eq 2: vitamin C dynamics
    dy[1] = (
        p["kin_vitc"] * (VitC0 - VitC)
        - p["kc_ne"] * max(NE - 1.0, 0.0) * VitC / VitC0
        - p["kc_vitc"] * max(Ctot - p["Cb_vitc"], 0.0) * VitC / VitC0
    )

    # Eq 3: aldosterone dynamics
    ald_drive = p["ka_ald"] * max(Ctot - p["th_ald"], 0.0) * (p["KMg_ald"] / (p["KMg_ald"] + max(Mg, 1e-12)))
    dy[2] = ald_drive - p["kcl_ald"] * (Ald - Ald0)

    # Eq 4: magnesium dynamics
    dy[3] = p["kin_mg"] * (Mg0 - Mg) - p["kw_mg"] * max(Ald - Ald0, 0.0) * max(Mg - p["Mg_floor"], 0.0)

    # Eq 5: BDNF dynamics
    dy[4] = p["krec_bdnf"] * (BDNF0 - BDNF) - p["ksil_bdnf"] * max(Ctot - p["K_bdnf"], 0.0) * BDNF / 100.0

    # Eq 6: Nrf2 dynamics
    dy[5] = p["krec_nrf"] * (NRF0 - Nrf2) - p["ksil_nrf"] * max(Ctot - p["K_nrf"], 0.0) * Nrf2 / 100.0

    # Eq 7: inflammation dynamics
    c_sustained = C0 + Cstress + float(ph.get("extra", 0.0))
    dy[6] = p["kinf"] * max(c_sustained - p["th_inf"], 0.0) * (1.0 + p["kauto"] * INF) - p["kcl_inf"] * INF

    # Eq 8 and Eq 9: tryptophan and kynurenine
    ido = p["kido"] * INF * Trp
    dy[7] = p["kin_trp"] * (TRP0 - Trp) - ido
    dy[8] = ido + p["kyn_base"] - p["kcl_kyn"] * Kyn

    # Eq 10: slow stress-load dynamics
    acc = (
        float(ph["acc"]) * p["kacc"] * psi_multiplier(str(ph["psi"]), BDNF, dep_gain, ptsd_gain)
        * max(Ctot - p["th_acc"], 0.0) * (1.0 - Cstress / p["Cmax"])
    )
    dy[9] = acc - p["kdec"] * Cstress
    return dy

def initial_state(ph: Dict[str, float | str], p: Dict[str, float], cstress_override: Optional[float] = None) -> np.ndarray:
    cstress0 = float(ph["Cstr0"] if cstress_override is None else cstress_override)
    return np.array([NE0, VitC0, Ald0, Mg0, BDNF0, NRF0, 0.0, TRP0, p["kyn_base"] / p["kcl_kyn"], cstress0], dtype=float)


def simulate(
    name: str,
    t_end: float = 168.0,
    n: int = 2000,
    parameter_overrides: Optional[Dict[str, float]] = None,
    phenotype_overrides: Optional[Dict[str, float | str]] = None,
    cstress0_override: Optional[float] = None,
    dep_gain: float = 0.60,
    ptsd_gain: float = 0.15,
):
    """Integrate one phenotype and return the scipy solution object.

    The returned object includes an additional ``Ctot`` array with total cortisol.
    """
    p = parameters(parameter_overrides)
    ph = phenotype(name, phenotype_overrides)
    teval = np.linspace(0.0, float(t_end), int(n))
    y0 = initial_state(ph, p, cstress0_override)
    sol = solve_ivp(
        rhs,
        (0.0, float(t_end)),
        y0,
        t_eval=teval,
        args=(ph, p, dep_gain, ptsd_gain),
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
        max_step=0.25,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed for {name}: {sol.message}")
    sol.Ctot = np.array([c_circadian(t, ph, p) + sol.y[IDX["Cstress"], i] + c_acute(t, ph) + float(ph.get("extra", 0.0)) for i, t in enumerate(sol.t)])
    sol.parameters = p
    sol.phenotype = ph
    return sol


def kyntrp(sol) -> np.ndarray:
    return sol.y[IDX["Kyn"]] / np.clip(sol.y[IDX["Trp"]], 1e-6, None)


def cross_time(sol, arr: np.ndarray, threshold: float, below: bool = True) -> Optional[float]:
    condition = arr < threshold if below else arr > threshold
    idx = np.where(condition)[0]
    return float(sol.t[idx[0]]) if len(idx) else None


def phenotype_summary(names: Iterable[str] = PHENOTYPE_ORDER) -> list[dict]:
    rows = []
    for name in names:
        sol = simulate(name)
        kt = kyntrp(sol)
        rows.append({
            "phenotype": name,
            "NE_day7": float(sol.y[IDX["NE"], -1]),
            "VitC_day7_uM": float(sol.y[IDX["VitC"], -1]),
            "Mg_day7_mmol_L": float(sol.y[IDX["Mg"], -1]),
            "BDNF_day7_percent": float(sol.y[IDX["BDNF"], -1]),
            "Nrf2_day7_percent": float(sol.y[IDX["Nrf2"], -1]),
            "INF_day7_au": float(sol.y[IDX["INF"], -1]),
            "KYN_TRP_day7": float(kt[-1]),
            "Cortisol_peak_ug_dL": float(sol.Ctot.max()),
            "Cortisol_mean_ug_dL": float(sol.Ctot.mean()),
            "Cortisol_nadir_ug_dL": float(sol.Ctot.min()),
        })
    return rows


def threshold_summary(names: Iterable[str] = ("chronic", "depressed")) -> list[dict]:
    rows = []
    for name in names:
        sol = simulate(name)
        kt = kyntrp(sol)
        rows.append({
            "phenotype": name,
            "Mg_lt_0_65_h": cross_time(sol, sol.y[IDX["Mg"]], THRESHOLDS["Mg_hypo"]),
            "VitC_lt_23_h": cross_time(sol, sol.y[IDX["VitC"]], THRESHOLDS["VitC_def"]),
            "BDNF_lt_70_h": cross_time(sol, sol.y[IDX["BDNF"]], THRESHOLDS["BDNF_70"]),
            "KYN_TRP_gt_0_08_h": cross_time(sol, kt, THRESHOLDS["KYNTRP_boundary"], below=False),
        })
    return rows


if __name__ == "__main__":
    print(f"{'phenotype':12s}{'NE168':>8s}{'VitC':>8s}{'Mg':>8s}{'BDNF':>8s}{'Nrf2':>8s}{'INF':>8s}{'KT':>8s}")
    for row in phenotype_summary():
        print(f"{row['phenotype']:12s}{row['NE_day7']:8.2f}{row['VitC_day7_uM']:8.1f}{row['Mg_day7_mmol_L']:8.2f}"
              f"{row['BDNF_day7_percent']:8.1f}{row['Nrf2_day7_percent']:8.1f}{row['INF_day7_au']:8.2f}{row['KYN_TRP_day7']:8.3f}")
    print("\nThreshold crossings (hours):")
    for row in threshold_summary():
        print(row)
