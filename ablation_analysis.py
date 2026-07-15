"""Structural ablation analysis for the Tiered Stress Biochemistry Model."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsbm_model as M

ABLATIONS = [
    {
        "scenario": "baseline_depressed",
        "phenotype": "depressed",
        "parameter_overrides": {},
        "phenotype_overrides": {},
        "description": "Default depression-like phenotype.",
    },
    {
        "scenario": "no_bdnf_feedback",
        "phenotype": "depressed",
        "parameter_overrides": {},
        "phenotype_overrides": {"psi": "none"},
        "description": "Depression-like phenotype with BDNF-dependent gain disabled.",
    },
    {
        "scenario": "no_slow_stress_accumulation",
        "phenotype": "depressed",
        "parameter_overrides": {"kacc": 0.0},
        "phenotype_overrides": {},
        "description": "Slow stress-load accumulation removed.",
    },
    {
        "scenario": "no_NE_vitaminC_consumption",
        "phenotype": "depressed",
        "parameter_overrides": {"kc_ne": 0.0},
        "phenotype_overrides": {},
        "description": "Fast noradrenergic/DBH vitamin C consumption removed.",
    },
    {
        "scenario": "no_inflammatory_self_amplification",
        "phenotype": "depressed",
        "parameter_overrides": {"kauto": 0.0},
        "phenotype_overrides": {},
        "description": "Inflammatory self-amplification removed.",
    },
    {
        "scenario": "no_aldosterone_Mg_wasting",
        "phenotype": "depressed",
        "parameter_overrides": {"kw_mg": 0.0},
        "phenotype_overrides": {},
        "description": "Aldosterone-driven magnesium wasting removed.",
    },
    {
        "scenario": "ptsd_with_depression_feedback",
        "phenotype": "ptsd",
        "parameter_overrides": {},
        "phenotype_overrides": {"acc": 1.0, "Cstr0": 3.0, "psi": "dep"},
        "description": "PTSD circadian baseline but depression-like slow stress-load accumulation and feedback sign.",
    },
]


def evaluate_scenario(item: dict) -> dict:
    sol = M.simulate(
        item["phenotype"],
        parameter_overrides=item["parameter_overrides"],
        phenotype_overrides=item["phenotype_overrides"],
    )
    kt = M.kyntrp(sol)
    return {
        "scenario": item["scenario"],
        "description": item["description"],
        "VitC_day7_uM": float(sol.y[M.IDX["VitC"], -1]),
        "Mg_day7_mmol_L": float(sol.y[M.IDX["Mg"], -1]),
        "BDNF_day7_percent": float(sol.y[M.IDX["BDNF"], -1]),
        "Nrf2_day7_percent": float(sol.y[M.IDX["Nrf2"], -1]),
        "INF_day7_au": float(sol.y[M.IDX["INF"], -1]),
        "KYN_TRP_day7": float(kt[-1]),
        "Mg_cross_h": M.cross_time(sol, sol.y[M.IDX["Mg"]], M.THRESHOLDS["Mg_hypo"]),
        "BDNF_cross_h": M.cross_time(sol, sol.y[M.IDX["BDNF"]], M.THRESHOLDS["BDNF_70"]),
        "KYN_TRP_cross_h": M.cross_time(sol, kt, M.THRESHOLDS["KYNTRP_boundary"], below=False),
    }


def save_ablation_figure(rows: list[dict], outpath: Path) -> None:
    labels = [r["scenario"].replace("_", "\n") for r in rows]
    bdnf = [r["BDNF_day7_percent"] for r in rows]
    vitc = [r["VitC_day7_uM"] for r in rows]
    kt = [r["KYN_TRP_day7"] * 300 for r in rows]  # scaled for shared visual axis
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(11.5, 5))
    ax.plot(x, bdnf, marker="o", label="BDNF (%)")
    ax.plot(x, vitc, marker="o", label="Vitamin C (µM)")
    ax.plot(x, kt, marker="o", label="KYN/TRP x300")
    ax.axhline(70, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(list(x), labels=labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("scaled day-7 output")
    ax.set_title("Ablation analysis: structural drivers of the depression-like outcome")
    ax.grid(axis="y", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=300, bbox_inches="tight"); plt.close(fig)


def run(output_dir: str | Path = "outputs") -> list[dict]:
    out = Path(output_dir)
    tables = out / "tables"; figs = out / "figures"
    tables.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_scenario(item) for item in ABLATIONS]
    with open(tables / "ablation_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with open(tables / "ablation_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    save_ablation_figure(rows, figs / "Ablation_analysis.png")
    print("Ablation analysis written to outputs/tables and outputs/figures.")
    return rows


def get_gui_spec():
    """Return the Tkinter task definition used by this script and the master launcher."""
    from tsbm_gui import TaskSpec

    def runner(output_dir: Path, options: dict) -> None:
        rows = run(output_dir)
        print(f"Completed {len(rows)} structural-ablation scenarios.")

    return TaskSpec(
        title="TSBM Structural Ablation Analysis",
        description=(
            "Runs the prespecified structural-ablation scenarios, writes CSV and JSON result tables, "
            "and generates the ablation comparison figure. The generated image is shown in the gallery "
            "and can be saved or opened in the basic editor."
        ),
        runner=runner,
        default_output="outputs",
        run_button_text="Run ablation analysis",
    )


if __name__ == "__main__":
    import argparse
    import sys

    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(description="Run TSBM structural ablation analysis.")
        parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--output", default="outputs")
        args = parser.parse_args()
        run(Path(args.output))
    else:
        from tsbm_gui import launch_task_gui
        launch_task_gui(get_gui_spec())
