"""Rigor analyses for TSBM: stability, local sensitivity, global robustness, and consistency checks."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import platform
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import tsbm_model as M

KEY_PARAMETERS = [
    "k_ne_on", "kc_ne", "kc_vitc", "kw_mg", "ksil_bdnf", "K_bdnf", "ksil_nrf",
    "kinf", "kauto", "kido", "kacc", "kdec", "Cmax", "Mg_floor",
]


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fast_final(name: str, parameter_overrides=None, phenotype_overrides=None, cstress0_override=None, dt: float = 0.5) -> tuple[np.ndarray, float]:
    """Return final state and final total cortisol using a fixed-step RK4 integrator.

    The manuscript simulations use scipy RK45. This RK4 helper is used only for
    repeated sensitivity/robustness scans to avoid thousands of adaptive-solver
    allocations. With dt=0.5 h it reproduces the qualitative ranking used in sensitivity and robustness scans while keeping the support package fast to rerun. Main manuscript values are generated with the RK45 solver in tsbm_model.py.
    """
    p = M.parameters(parameter_overrides)
    ph = M.phenotype(name, phenotype_overrides)
    y = M.initial_state(ph, p, cstress0_override)
    t = 0.0
    t_end = 168.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = M.rhs(t, y, ph, p, 0.60, 0.15)
        k2 = M.rhs(t + 0.5*h, y + 0.5*h*k1, ph, p, 0.60, 0.15)
        k3 = M.rhs(t + 0.5*h, y + 0.5*h*k2, ph, p, 0.60, 0.15)
        k4 = M.rhs(t + h, y + h*k3, ph, p, 0.60, 0.15)
        y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    ctot = M.c_circadian(t_end, ph, p) + y[M.IDX["Cstress"]] + M.c_acute(t_end, ph) + float(ph.get("extra", 0.0))
    return y, float(ctot)

def summary_outputs(parameter_overrides=None, phenotype_overrides=None, include_cross: bool = False) -> dict:
    dep_y, _ = fast_final("depressed", parameter_overrides=parameter_overrides, phenotype_overrides=phenotype_overrides)
    ptsd_y, _ = fast_final("ptsd", parameter_overrides=parameter_overrides)
    chronic_y, _ = fast_final("chronic", parameter_overrides=parameter_overrides)
    loads = np.linspace(0, 12, 13)
    bdnf = np.array([
        fast_final("chronic", parameter_overrides=parameter_overrides, cstress0_override=float(L), phenotype_overrides={"psi": "dep"})[0][M.IDX["BDNF"]]
        for L in loads
    ])
    below = np.where(bdnf < 60)[0]
    tipping = float(np.interp(60.0, bdnf[::-1], loads[::-1])) if len(below) else 12.0
    dep_trp = max(dep_y[M.IDX["Trp"]], 1e-6)
    out = {
        "dep_BDNF": float(dep_y[M.IDX["BDNF"]]),
        "ptsd_BDNF": float(ptsd_y[M.IDX["BDNF"]]),
        "chronic_BDNF": float(chronic_y[M.IDX["BDNF"]]),
        "dep_ptsd_BDNF_gap": float(ptsd_y[M.IDX["BDNF"]] - dep_y[M.IDX["BDNF"]]),
        "dep_Mg_cross_h": None,
        "dep_KYN_TRP": float(dep_y[M.IDX["Kyn"]] / dep_trp),
        "tipping_load": tipping,
    }
    if include_cross:
        dep = M.simulate("depressed", parameter_overrides=parameter_overrides, phenotype_overrides=phenotype_overrides, n=800)
        out["dep_Mg_cross_h"] = M.cross_time(dep, dep.y[M.IDX["Mg"]], 0.65)
    return out

def stability_analysis() -> dict:
    p = M.parameters()
    ph = M.phenotype("normal", {"C1": 0.0, "acc": 0.0})
    y0 = M.initial_state(ph, p)
    sol = solve_ivp(M.rhs, (0, 4000), y0, args=(ph, p, 0.60, 0.15), method="LSODA", rtol=1e-9, atol=1e-12)
    xstar = sol.y[:, -1]

    def f(x):
        return M.rhs(0.0, x, ph, p, 0.60, 0.15)

    residual = float(np.max(np.abs(f(xstar))))
    n = len(xstar)
    jac = np.zeros((n, n))
    h = 1e-6
    f0 = f(xstar)
    for j in range(n):
        xp = xstar.copy(); xp[j] += h
        jac[:, j] = (f(xp) - f0) / h
    eig = np.linalg.eigvals(jac)
    return {
        "n_states": n,
        "residual_at_fixed_point": residual,
        "max_real_eigenvalue": float(np.max(eig.real)),
        "stable_autonomous_unstressed_equilibrium": bool(np.max(eig.real) < 0),
        "eigenvalues_real": [float(x) for x in eig.real],
        "eigenvalues_imag": [float(x) for x in eig.imag],
    }


def local_sensitivity() -> list[dict]:
    """One-at-a-time ±20% sensitivity analysis.

    Cmax is evaluated first because some ODE solvers can spend longer when the
    saturation ceiling is changed after many preceding integrations in one process.
    """
    base = summary_outputs(include_cross=True)
    rows = []
    ordered = ["Cmax"] + [p for p in KEY_PARAMETERS if p != "Cmax"]
    for param in ordered:
        value = M.DEFAULT_PARAMETERS[param]
        lo = summary_outputs({param: value * 0.8})
        hi = summary_outputs({param: value * 1.2})
        row = {
            "parameter": param,
            "BDNF_at_minus20": lo["dep_BDNF"],
            "BDNF_at_plus20": hi["dep_BDNF"],
            "BDNF_span": abs(hi["dep_BDNF"] - lo["dep_BDNF"]),
            "tipping_at_minus20": lo["tipping_load"],
            "tipping_at_plus20": hi["tipping_load"],
            "tipping_span": abs(hi["tipping_load"] - lo["tipping_load"]),
            "base_BDNF": base["dep_BDNF"],
            "base_tipping": base["tipping_load"],
        }
        rows.append(row)
    return sorted(rows, key=lambda row: row["BDNF_span"], reverse=True)

def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    sample = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        sample[:, j] = (perm + rng.random(n)) / n
    return sample


def _state_outputs(name: str, parameter_overrides=None, phenotype_overrides=None) -> dict:
    y, _ = fast_final(name, parameter_overrides=parameter_overrides, phenotype_overrides=phenotype_overrides)
    trp = max(float(y[M.IDX["Trp"]]), 1e-9)
    return {
        "VitC": float(y[M.IDX["VitC"]]),
        "Mg": float(y[M.IDX["Mg"]]),
        "BDNF": float(y[M.IDX["BDNF"]]),
        "Nrf2": float(y[M.IDX["Nrf2"]]),
        "INF": float(y[M.IDX["INF"]]),
        "KYN_TRP": float(y[M.IDX["Kyn"]] / trp),
    }


def local_sensitivity_multioutput() -> list[dict]:
    """One-at-a-time ±20% sensitivity across six depression day-7 outputs.

    The normalized span is |Y(+20%)-Y(-20%)|/(0.4*|Y_baseline|).
    It is a screening metric, not a formal identifiability estimate.
    """
    base = _state_outputs("depressed")
    rows = []
    for param in KEY_PARAMETERS:
        value = M.DEFAULT_PARAMETERS[param]
        lo = _state_outputs("depressed", {param: value * 0.8})
        hi = _state_outputs("depressed", {param: value * 1.2})
        row = {"parameter": param}
        for output in ("VitC", "Mg", "BDNF", "Nrf2", "INF", "KYN_TRP"):
            span = abs(hi[output] - lo[output])
            denom = max(0.4 * abs(base[output]), 1e-12)
            row[f"{output}_baseline"] = base[output]
            row[f"{output}_minus20"] = lo[output]
            row[f"{output}_plus20"] = hi[output]
            row[f"{output}_normalized_span"] = span / denom
        rows.append(row)
    return rows


def global_robustness(n: int = 400, seed: int = 20240101) -> dict:
    """Latin-hypercube robustness scan over independent ±20% parameter ranges.

    Two prespecified criteria are reported. The legacy BDNF-only criterion is
    retained for continuity. A stricter multi-output criterion additionally
    requires depression-like magnesium and KYN/TRP threshold crossing, PTSD-like
    preservation of those outputs, and the BDNF ordering depression < chronic <
    PTSD.
    """
    rng = np.random.default_rng(seed)
    lhs = latin_hypercube(n, len(KEY_PARAMETERS), rng)
    scales = 0.8 + 0.4 * lhs
    ok_bdnf = 0
    ok_multi = 0
    rows = []
    for i in range(n):
        overrides = {param: M.DEFAULT_PARAMETERS[param] * scales[i, j] for j, param in enumerate(KEY_PARAMETERS)}
        dep = _state_outputs("depressed", overrides)
        ptsd = _state_outputs("ptsd", overrides)
        chronic = _state_outputs("chronic", overrides)
        bdnf_condition = (dep["BDNF"] < 70.0) and (ptsd["BDNF"] > 85.0) and ((ptsd["BDNF"] - dep["BDNF"]) > 20.0)
        multi_condition = (
            bdnf_condition
            and dep["Mg"] < 0.65
            and dep["KYN_TRP"] > 0.08
            and ptsd["Mg"] > 0.70
            and ptsd["KYN_TRP"] < 0.08
            and dep["BDNF"] < chronic["BDNF"] < ptsd["BDNF"]
        )
        ok_bdnf += int(bdnf_condition)
        ok_multi += int(multi_condition)
        row = {"draw": i + 1, "bdnf_only_ok": bool(bdnf_condition), "multi_output_ok": bool(multi_condition)}
        for prefix, vals in (("dep", dep), ("ptsd", ptsd), ("chronic", chronic)):
            for key, value in vals.items():
                row[f"{prefix}_{key}"] = float(value)
        rows.append(row)
    return {
        "n": n,
        "seed": seed,
        "method": "Latin hypercube, independent ±20% ranges",
        "bdnf_only_fraction": ok_bdnf / n,
        "multi_output_fraction": ok_multi / n,
        "criteria": {
            "bdnf_only": "depression BDNF <70%, PTSD BDNF >85%, PTSD-depression gap >20 percentage points",
            "multi_output": "BDNF-only criterion plus depression Mg <0.65 mmol/L and KYN/TRP >0.08; PTSD Mg >0.70 mmol/L and KYN/TRP <0.08; depression < chronic < PTSD BDNF",
        },
        "draws": rows,
    }


def consistency_table() -> list[dict]:
    normal = M.simulate("normal")
    dep = M.simulate("depressed")
    return [
        {"Quantity": "Normal cortisol peak (ug/dL)", "Literature": "~17.5 (morning)", "Model": f"{normal.Ctot.max():.1f}", "Type": "calibration", "Source": "Yehuda et al. (1996)"},
        {"Quantity": "Normal cortisol nadir (ug/dL)", "Literature": "~2-3", "Model": f"{normal.Ctot.min():.1f}", "Type": "calibration", "Source": "Yehuda et al. (1996)"},
        {"Quantity": "Normal mean cortisol (ug/dL)", "Literature": "~10-12", "Model": f"{normal.Ctot.mean():.1f}", "Type": "calibration", "Source": "Yehuda et al. (1996)"},
        {"Quantity": "Healthy KYN/TRP ratio", "Literature": "0.03-0.05", "Model": f"{M.kyntrp(normal)[-1]:.3f}", "Type": "calibration", "Source": "Maes et al. (2011)"},
        {"Quantity": "Model KYN/TRP reference boundary", "Literature": "0.08 (same-unit ratio; assay-dependent)", "Model": f"{M.kyntrp(dep)[-1]:.3f}", "Type": "calibration", "Source": "Maes et al. (2011); Reus et al. (2015)"},
        {"Quantity": "Crossing of model magnesium boundary", "Literature": "within 24 h (calibration target)", "Model": f"{M.cross_time(dep, dep.y[M.IDX['Mg']], 0.65):.0f} h", "Type": "calibration", "Source": "Golf et al. (1998)"},
        {"Quantity": "Normal plasma vitamin C (uM)", "Literature": "50-70", "Model": f"{normal.y[M.IDX['VitC'], -1]:.0f}", "Type": "calibration", "Source": "Levine et al. (1996)"},
        {"Quantity": "BDNF suppression (depression)", "Literature": "40-60% reduction", "Model": f"{100 - dep.y[M.IDX['BDNF'], -1]:.0f}% reduction", "Type": "consistency", "Source": "Duman & Monteggia (2006); Tsankova et al. (2006)"},
    ]


def save_sensitivity_figure(rows: list[dict], outpath: Path) -> None:
    base_bdnf = rows[0]["base_BDNF"]
    base_tip = rows[0]["base_tipping"]
    labels = [r["parameter"] for r in rows]
    y = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for i, r in enumerate(rows):
        axes[0].plot([r["BDNF_at_minus20"], r["BDNF_at_plus20"]], [i, i], lw=6, solid_capstyle="round", color="#3a5a99", alpha=0.85)
        axes[1].plot([r["tipping_at_minus20"], r["tipping_at_plus20"]], [i, i], lw=6, solid_capstyle="round", color="#c1440e", alpha=0.85)
    axes[0].axvline(base_bdnf, color="black", ls=":", lw=1)
    axes[1].axvline(base_tip, color="black", ls=":", lw=1)
    for ax in axes:
        ax.set_yticks(y, labels=labels, fontsize=8)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.25)
    axes[0].set_xlabel("depression day-7 BDNF (%)"); axes[0].set_title("Figure 8a. Sensitivity of depression BDNF")
    axes[1].set_xlabel("operational threshold load (ug/dL)"); axes[1].set_title("Figure 8b. Sensitivity of operational threshold")
    fig.tight_layout(); fig.savefig(outpath, dpi=300, bbox_inches="tight"); plt.close(fig)


def run(output_dir: str | Path = "outputs") -> dict:
    out = Path(output_dir)
    tables = out / "tables"; figs = out / "figures"
    _ensure(tables); _ensure(figs)
    base = summary_outputs(include_cross=True)
    stability = stability_analysis()
    # Run global robustness before local sensitivity to avoid adaptive-solver resource buildup.
    robustness = global_robustness(n=100)
    sensitivity = local_sensitivity()
    consistency = consistency_table()

    with open(tables / "sensitivity_local.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        writer.writeheader(); writer.writerows(sensitivity)
    with open(tables / "validation_consistency_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(consistency[0].keys()))
        writer.writeheader(); writer.writerows(consistency)
    with open(tables / "global_robustness_draws.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(robustness["draws"][0].keys()))
        writer.writeheader(); writer.writerows(robustness["draws"])
    save_sensitivity_figure(sensitivity, figs / "Figure_8_sensitivity.png")

    result = {
        "base": base,
        "stability": stability,
        "robustness": {k: v for k, v in robustness.items() if k != "draws"},
        "sensitivity": sensitivity,
        "consistency_table": consistency,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": "RK45",
            "rtol": "1e-6",
            "atol": "1e-9",
            "max_step_h": 0.25,
        },
    }
    with open(tables / "tsbm_rigor.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Stability max Re(eig) = {stability['max_real_eigenvalue']:.4g}; multi-output robustness = {robustness['multi_output_fraction']*100:.1f}%")
    return result


# ---------- Separate writers used by run_all.py to avoid long single-process scans ----------
def write_base_outputs(output_dir: str | Path = "outputs") -> dict:
    out = Path(output_dir); tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    base = summary_outputs(include_cross=True)
    stability = stability_analysis()
    consistency = consistency_table()
    with open(tables / "validation_consistency_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(consistency[0].keys()))
        writer.writeheader(); writer.writerows(consistency)
    result = {
        "base": base,
        "stability": stability,
        "consistency_table": consistency,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "solver": "RK45 for main simulations; RK4 helper for repeated sensitivity scans",
            "rtol": "1e-6",
            "atol": "1e-9",
            "max_step_h": 0.25,
        },
    }
    with open(tables / "tsbm_rigor_base.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Base/stability outputs written. Max Re(eig) = {stability['max_real_eigenvalue']:.4g}")
    return result


def write_sensitivity_outputs(output_dir: str | Path = "outputs") -> list[dict]:
    out = Path(output_dir); tables = out / "tables"; figs = out / "figures"
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    sensitivity = local_sensitivity()
    multi = local_sensitivity_multioutput()
    with open(tables / "sensitivity_local.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensitivity[0].keys()))
        writer.writeheader(); writer.writerows(sensitivity)
    with open(tables / "sensitivity_local.json", "w") as f:
        json.dump(sensitivity, f, indent=2)
    with open(tables / "sensitivity_local_multioutput.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(multi[0].keys()))
        writer.writeheader(); writer.writerows(multi)
    with open(tables / "sensitivity_local_multioutput.json", "w") as f:
        json.dump(multi, f, indent=2)
    save_sensitivity_figure(sensitivity, figs / "Figure_8_sensitivity.png")
    print("Sensitivity outputs written, including multi-output screening.")
    return sensitivity


def _distribution_stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q2_5": float(np.percentile(arr, 2.5)),
        "median": float(np.percentile(arr, 50)),
        "q97_5": float(np.percentile(arr, 97.5)),
        "max": float(np.max(arr)),
    }


def write_robustness_outputs(output_dir: str | Path = "outputs", n: int = 400) -> dict:
    out = Path(output_dir); tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    robustness = global_robustness(n=n)
    with open(tables / "global_robustness_draws.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(robustness["draws"][0].keys()))
        writer.writeheader(); writer.writerows(robustness["draws"])
    distribution = {}
    for prefix in ("dep", "ptsd", "chronic"):
        for output in ("VitC", "Mg", "BDNF", "Nrf2", "INF", "KYN_TRP"):
            key = f"{prefix}_{output}"
            distribution[key] = _distribution_stats([r[key] for r in robustness["draws"]])
    with open(tables / "global_robustness_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)
    with open(tables / "global_robustness_distribution.csv", "w", newline="") as f:
        fieldnames = ["output", "mean", "sd", "min", "q2_5", "median", "q97_5", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for output, stats in distribution.items():
            row = {"output": output}; row.update(stats); writer.writerow(row)
    summary = {k: v for k, v in robustness.items() if k != "draws"}
    summary["distribution"] = distribution
    with open(tables / "global_robustness.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"Robustness outputs written: BDNF-only {summary['bdnf_only_fraction']*100:.1f}%; "
        f"multi-output {summary['multi_output_fraction']*100:.1f}% of {n} draws."
    )
    return summary


def combine_rigor_outputs(output_dir: str | Path = "outputs") -> dict:
    out = Path(output_dir); tables = out / "tables"
    with open(tables / "tsbm_rigor_base.json") as f:
        base = json.load(f)
    with open(tables / "global_robustness.json") as f:
        robustness = json.load(f)
    with open(tables / "sensitivity_local.json") as f:
        sensitivity = json.load(f)
    combined = dict(base)
    combined["robustness"] = robustness
    combined["sensitivity"] = sensitivity
    with open(tables / "tsbm_rigor.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("Combined rigor JSON written.")
    return combined


def get_gui_spec():
    """Return the Tkinter task definition used by this script and the master launcher."""
    from tsbm_gui import OptionSpec, TaskSpec

    def runner(output_dir: Path, options: dict) -> None:
        part = str(options["part"])
        n = int(options["n"])
        if n < 2:
            raise ValueError("The number of robustness draws must be at least 2.")
        if part == "Base checks":
            write_base_outputs(output_dir)
        elif part == "Local sensitivity":
            write_sensitivity_outputs(output_dir)
        elif part == "Global robustness":
            write_robustness_outputs(output_dir, n=n)
        elif part == "Combine existing outputs":
            combine_rigor_outputs(output_dir)
        else:
            print("Running base, robustness, sensitivity, and combined outputs...")
            write_base_outputs(output_dir)
            write_robustness_outputs(output_dir, n=n)
            write_sensitivity_outputs(output_dir)
            combine_rigor_outputs(output_dir)

    return TaskSpec(
        title="TSBM Rigor and Robustness Analyses",
        description=(
            "Runs the autonomous stability check, calibration-consistency table, local sensitivity analysis, "
            "multi-output sensitivity analysis, and Latin-hypercube robustness analysis. It generates CSV/JSON "
            "tables and the sensitivity figure."
        ),
        runner=runner,
        default_output="outputs",
        options=(
            OptionSpec(
                key="part", label="Analysis section", kind="choice", default="All analyses",
                choices=("All analyses", "Base checks", "Local sensitivity", "Global robustness", "Combine existing outputs"),
            ),
            OptionSpec(
                key="n", label="Robustness draws", kind="int", default=400,
                help_text="Used only for global robustness. Use 20-50 for a quick test.",
            ),
        ),
        run_button_text="Run rigor analysis",
        warning="A full 400-draw robustness run may take several minutes.",
    )


if __name__ == "__main__":
    import argparse
    import sys

    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(description="Run TSBM rigor analyses.")
        parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--part", choices=["base", "sensitivity", "robustness", "combine", "all"], default="all")
        parser.add_argument("--output", default="outputs")
        parser.add_argument("--n", type=int, default=400, help="Number of global robustness draws")
        args = parser.parse_args()
        if args.part == "base":
            write_base_outputs(args.output)
        elif args.part == "sensitivity":
            write_sensitivity_outputs(args.output)
        elif args.part == "robustness":
            write_robustness_outputs(args.output, n=args.n)
        elif args.part == "combine":
            combine_rigor_outputs(args.output)
        else:
            write_base_outputs(args.output)
            write_robustness_outputs(args.output, n=args.n)
            write_sensitivity_outputs(args.output)
            combine_rigor_outputs(args.output)
    else:
        from tsbm_gui import launch_task_gui
        launch_task_gui(get_gui_spec())
