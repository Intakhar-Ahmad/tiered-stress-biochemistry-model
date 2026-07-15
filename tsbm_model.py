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


def export_simulation(
    output_dir: str | Path = "model_outputs",
    scenario: str = "depressed",
    t_end: float = 168.0,
    n: int = 2000,
) -> dict:
    """Run one scenario and export a trajectory CSV, summary JSON, and overview figure."""
    import csv
    import json
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if scenario not in PHENOTYPE_ORDER:
        raise ValueError(f"Unknown scenario {scenario!r}. Choose from {PHENOTYPE_ORDER}.")
    if t_end <= 0:
        raise ValueError("Simulation horizon must be greater than zero.")
    if n < 20:
        raise ValueError("Time points must be at least 20.")

    out = Path(output_dir)
    tables = out / "tables"
    figures = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    sol = simulate(scenario, t_end=t_end, n=n)
    ratio = kyntrp(sol)
    csv_path = tables / f"trajectory_{scenario}.csv"
    fieldnames = ["time_h", *STATES, "Ctotal", "KYN_TRP"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, time_h in enumerate(sol.t):
            row = {"time_h": float(time_h), "Ctotal": float(sol.Ctot[i]), "KYN_TRP": float(ratio[i])}
            row.update({state: float(sol.y[IDX[state], i]) for state in STATES})
            writer.writerow(row)

    summary = {
        "scenario": scenario,
        "horizon_h": float(t_end),
        "time_points": int(n),
        "day_end": {state: float(sol.y[IDX[state], -1]) for state in STATES},
        "Ctotal_peak": float(np.max(sol.Ctot)),
        "Ctotal_mean": float(np.mean(sol.Ctot)),
        "Ctotal_nadir": float(np.min(sol.Ctot)),
        "KYN_TRP_end": float(ratio[-1]),
        "threshold_crossings_h": {
            "Mg_lt_0_65": cross_time(sol, sol.y[IDX["Mg"]], THRESHOLDS["Mg_hypo"]),
            "VitC_lt_23": cross_time(sol, sol.y[IDX["VitC"]], THRESHOLDS["VitC_def"]),
            "BDNF_lt_70": cross_time(sol, sol.y[IDX["BDNF"]], THRESHOLDS["BDNF_70"]),
            "KYN_TRP_gt_0_08": cross_time(sol, ratio, THRESHOLDS["KYNTRP_boundary"], below=False),
        },
    }
    with open(tables / f"summary_{scenario}.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    ax = axes[0, 0]
    ax.plot(sol.t, sol.Ctot, label="Total cortisol")
    ax.set_ylabel("Cortisol (µg/dL)")
    ax.set_title("Total model cortisol")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(sol.t, sol.y[IDX["VitC"]], label="Vitamin C")
    ax.plot(sol.t, sol.y[IDX["Mg"]] * 60, label="Magnesium ×60")
    ax.axhline(THRESHOLDS["VitC_def"], linestyle="--", linewidth=1, label="Vitamin C boundary")
    ax.set_ylabel("Scaled resource level")
    ax.set_title("Vitamin C and magnesium")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(sol.t, sol.y[IDX["BDNF"]], label="BDNF")
    ax.plot(sol.t, sol.y[IDX["Nrf2"]], label="Nrf2")
    ax.axhline(THRESHOLDS["BDNF_70"], linestyle="--", linewidth=1, label="BDNF 70% boundary")
    ax.set_ylabel("Normalized state (%)")
    ax.set_title("BDNF- and Nrf2-related states")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(sol.t, sol.y[IDX["INF"]], label="Inflammation")
    ax.plot(sol.t, ratio * 50, label="KYN/TRP ×50")
    ax.axhline(THRESHOLDS["KYNTRP_boundary"] * 50, linestyle="--", linewidth=1, label="KYN/TRP boundary ×50")
    ax.set_ylabel("Scaled output")
    ax.set_title("Inflammation and kynurenine shunt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    for ax in axes.ravel():
        ax.set_xlabel("Time (h)")
    fig.suptitle(f"TSBM scenario simulation: {scenario}", y=1.01)
    fig.tight_layout()
    figure_path = figures / f"Model_simulation_{scenario}.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Scenario: {scenario}")
    print(f"Trajectory table: {csv_path}")
    print(f"Summary table: {tables / f'summary_{scenario}.json'}")
    print(f"Figure: {figure_path}")
    return summary


def get_gui_spec():
    """Return the Tkinter task definition used by this script and the master launcher."""
    from tsbm_gui import OptionSpec, TaskSpec

    def runner(output_dir: Path, options: dict) -> None:
        export_simulation(
            output_dir=output_dir,
            scenario=str(options["scenario"]),
            t_end=float(options["t_end"]),
            n=int(options["n"]),
        )

    return TaskSpec(
        title="TSBM Model Simulator",
        description=(
            "Runs one of the five prespecified TSBM scenarios. The tool exports the complete time-course table, "
            "a JSON summary with threshold crossings, and a four-panel overview image. Generated images can be "
            "saved as copies or opened in the basic editor."
        ),
        runner=runner,
        default_output="model_outputs",
        options=(
            OptionSpec(key="scenario", label="Scenario", kind="choice", default="depressed", choices=tuple(PHENOTYPE_ORDER)),
            OptionSpec(key="t_end", label="Simulation horizon (hours)", kind="float", default=168.0),
            OptionSpec(key="n", label="Number of time points", kind="int", default=2000),
        ),
        run_button_text="Run simulation",
    )


if __name__ == "__main__":
    import argparse
    import sys

    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(description="Run one TSBM scenario and export outputs.")
        parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--output", default="model_outputs")
        parser.add_argument("--scenario", choices=PHENOTYPE_ORDER, default="depressed")
        parser.add_argument("--t-end", type=float, default=168.0)
        parser.add_argument("--n", type=int, default=2000)
        args = parser.parse_args()
        export_simulation(args.output, args.scenario, args.t_end, args.n)
    else:
        from tsbm_gui import launch_task_gui
        launch_task_gui(get_gui_spec())
