"""Generate all manuscript figures for the Tiered Stress Biochemistry Model."""
from __future__ import annotations

from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tsbm_model as M

COLORS = {
    "normal": "#2e8b57",
    "acute": "#e0a458",
    "chronic": "#3a5a99",
    "depressed": "#c1440e",
    "ptsd": "#6a4c93",
}


def _ensure(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _all_solutions():
    return {name: M.simulate(name) for name in M.PHENOTYPE_ORDER}


def export_tables(outdir: Path) -> None:
    _ensure(outdir)
    with open(outdir / "phenotype_day7_summary.csv", "w", newline="") as f:
        rows = M.phenotype_summary()
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with open(outdir / "threshold_crossing_times.csv", "w", newline="") as f:
        rows = M.threshold_summary()
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    # Machine-readable parameter table.
    param_rows = [
        {"parameter": k, "value": v, "notes": PARAMETER_NOTES.get(k, "")}
        for k, v in M.DEFAULT_PARAMETERS.items()
    ]
    with open(outdir / "parameters.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parameter", "value", "notes"])
        writer.writeheader(); writer.writerows(param_rows)


PARAMETER_NOTES = {
    "t_peak": "h; literature-derived morning cortisol peak",
    "k_ne_on": "scaled h^-1; literature-constrained/calibrated noradrenergic synthesis gain",
    "k_ne_off": "h^-1; literature-constrained fast clearance",
    "k_thind": "per ug/dL; literature-constrained/calibrated TH chronic induction",
    "kin_vitc": "h^-1; literature-constrained by vitamin C turnover",
    "kc_vitc": "h^-1 per ug/dL; calibrated cortisol/oxidative consumption",
    "kc_ne": "h^-1 scaled; calibrated DBH/noradrenergic consumption",
    "Cb_vitc": "ug/dL; cortisol threshold for vitamin C consumption",
    "ka_ald": "scaled activation; calibrated/literature-constrained",
    "kcl_ald": "h^-1; aldosterone clearance toward baseline",
    "th_ald": "ug/dL; aldosterone activation threshold",
    "KMg_ald": "mmol/L; Mg sensitivity constant",
    "kin_mg": "h^-1; Mg recovery",
    "kw_mg": "h^-1 per aldosterone unit; Mg wasting",
    "Mg_floor": "mmol/L; lower physiologic floor for the serum-comparable exchangeable Mg proxy",
    "krec_bdnf": "h^-1; BDNF recovery",
    "ksil_bdnf": "h^-1 per ug/dL; BDNF silencing",
    "K_bdnf": "ug/dL; BDNF suppression threshold",
    "krec_nrf": "h^-1; Nrf2 recovery",
    "ksil_nrf": "h^-1 per ug/dL; Nrf2 suppression",
    "K_nrf": "ug/dL; Nrf2 suppression threshold",
    "kinf": "a.u. h^-1; inflammation activation",
    "kauto": "a.u.^-1; inflammatory self-amplification",
    "kcl_inf": "h^-1; inflammation clearance",
    "th_inf": "ug/dL; inflammation threshold",
    "kin_trp": "h^-1; tryptophan recovery",
    "kido": "uM^-1 h^-1; IDO conversion",
    "kcl_kyn": "h^-1; kynurenine clearance",
    "kyn_base": "uM h^-1; basal kynurenine production",
    "kacc": "scaled h^-1; slow stress-load accumulation",
    "kdec": "h^-1; slow stress-load decay",
    "th_acc": "ug/dL; accumulation threshold",
    "Cmax": "ug/dL; saturation ceiling",
}


def figure2_day7_heatmap(outdir: Path) -> None:
    sols = _all_solutions()
    metrics = ["NE", "VitC", "Mg", "BDNF", "Nrf2", "INF", "KYN/TRP"]
    raw = []
    health = []
    for name in M.PHENOTYPE_ORDER:
        sol = sols[name]
        kt = M.kyntrp(sol)[-1]
        vals = [sol.y[M.IDX["NE"], -1], sol.y[M.IDX["VitC"], -1], sol.y[M.IDX["Mg"], -1], sol.y[M.IDX["BDNF"], -1], sol.y[M.IDX["Nrf2"], -1], sol.y[M.IDX["INF"], -1], kt]
        raw.append(vals)
        # health index: preserved = 1; depleted/elevated = 0.
        health.append([
            max(0, min(1, 1 - abs(vals[0]-1)/1.0)),
            max(0, min(1, vals[1] / 60.0)),
            max(0, min(1, vals[2] / 0.85)),
            max(0, min(1, vals[3] / 100.0)),
            max(0, min(1, vals[4] / 100.0)),
            max(0, min(1, 1 - vals[5] / 15.0)),
            max(0, min(1, 1 - (vals[6] - 0.035) / 0.14)),
        ])
    raw = np.array(raw); health = np.array(health)
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    im = ax.imshow(health, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), labels=metrics)
    ax.set_yticks(range(len(M.PHENOTYPE_ORDER)), labels=[p.capitalize() for p in M.PHENOTYPE_ORDER])
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            text = f"{raw[i,j]:.2g}" if metrics[j] in ("NE", "Mg", "INF", "KYN/TRP") else f"{raw[i,j]:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    ax.set_title("Figure 2. Day-7 biochemical state across phenotypes")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("health index")
    _save(fig, outdir / "Figure_2_day7_heatmap.png")


def figure3_cortisol_backbone(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    t = np.linspace(0, 72, 1000)
    for name in ["normal", "depressed", "ptsd"]:
        ph = M.phenotype(name)
        p = M.parameters()
        c = [M.c_circadian(x, ph, p) for x in t]
        ax.plot(t, c, lw=2, label=name, color=COLORS[name])
    ax.set_xlabel("time (h)"); ax.set_ylabel("cortisol (µg/dL)")
    ax.set_title("Figure 3. Circadian cortisol backbone by phenotype")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, outdir / "Figure_3_cortisol_backbone.png")


def figure4_tier1(outdir: Path) -> None:
    sols = _all_solutions()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5))
    panels = [("VitC", "Vitamin C (µM)"), ("Mg", "Magnesium (mmol/L)"), ("Ald", "Aldosterone (a.u.)")]
    for ax, (key, ylabel) in zip(axes[0], panels):
        for name, sol in sols.items():
            ax.plot(sol.t, sol.y[M.IDX[key]], label=name, color=COLORS[name], lw=1.5)
        ax.set_title(key); ax.set_xlabel("time (h)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    # Detail panels: chronic/depressed vitamin C and Mg with thresholds.
    for name in ["chronic", "depressed"]:
        sol = sols[name]
        axes[1,0].plot(sol.t, sol.y[M.IDX["VitC"]], label=name, color=COLORS[name], lw=1.8)
        axes[1,1].plot(sol.t, sol.y[M.IDX["Mg"]], label=name, color=COLORS[name], lw=1.8)
        axes[1,2].plot(sol.t, M.kyntrp(sol), label=name, color=COLORS[name], lw=1.8)
    axes[1,0].axhline(M.THRESHOLDS["VitC_def"], ls="--", color="red", lw=1)
    axes[1,1].axhline(M.THRESHOLDS["Mg_hypo"], ls="--", color="red", lw=1)
    axes[1,2].axhline(M.THRESHOLDS["KYNTRP_boundary"], ls="--", color="red", lw=1)
    axes[1,0].set_title("Vitamin C threshold detail"); axes[1,0].set_ylabel("Vitamin C (µM)")
    axes[1,1].set_title("Magnesium threshold detail"); axes[1,1].set_ylabel("Magnesium (mmol/L)")
    axes[1,2].set_title("KYN/TRP ratio"); axes[1,2].set_ylabel("ratio")
    for ax in axes[1]:
        ax.set_xlabel("time (h)"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Figure 4. Tier 1 dynamics and kynurenine shunt", y=1.02)
    _save(fig, outdir / "Figure_4_tier1_dynamics.png")


def figure5_tier2(outdir: Path) -> None:
    sols = _all_solutions()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    panels = [("BDNF", "BDNF (%)"), ("Nrf2", "Nrf2 (%)"), ("INF", "Inflammation (a.u.)"), ("Ctot", "Total cortisol (µg/dL)")]
    for ax, (key, ylabel) in zip(axes.ravel(), panels):
        for name, sol in sols.items():
            arr = sol.Ctot if key == "Ctot" else sol.y[M.IDX[key]]
            ax.plot(sol.t, arr, label=name, color=COLORS[name], lw=1.5)
        ax.set_title(key); ax.set_xlabel("time (h)"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25)
    axes[0,0].axhline(70, ls="--", color="red", lw=1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Figure 5. Tier 2 reprogramming", y=1.02)
    _save(fig, outdir / "Figure_5_tier2_reprogramming.png")


def figure6_thresholds(outdir: Path) -> None:
    rows = M.threshold_summary()
    labels = ["Mg < 0.65", "BDNF < 70", "VitC < 23", "KYN/TRP > 0.08"]
    keys = ["Mg_lt_0_65_h", "BDNF_lt_70_h", "VitC_lt_23_h", "KYN_TRP_gt_0_08_h"]
    fig, ax = plt.subplots(figsize=(8, 4.7))
    ypos = []
    vals = []
    colors = []
    ylabels = []
    for r, row in enumerate(rows):
        for k, label in zip(keys, labels):
            val = row[k]
            ypos.append(len(ypos)); vals.append(np.nan if val is None else val)
            colors.append(COLORS[row["phenotype"]]); ylabels.append(f"{row['phenotype']}: {label}")
    ax.barh(ypos, np.nan_to_num(vals, nan=0.0), color=colors, alpha=0.85)
    for y, val in zip(ypos, vals):
        if np.isnan(val):
            ax.text(2, y, "not crossed", va="center", fontsize=8)
        else:
            ax.text(val + 1, y, f"{val:.1f} h", va="center", fontsize=8)
    ax.axvline(24, color="gray", ls=":", lw=1); ax.axvline(72, color="gray", ls=":", lw=1)
    ax.set_yticks(ypos, labels=ylabels); ax.invert_yaxis(); ax.set_xlabel("hours to threshold crossing")
    ax.set_title("Figure 6. Staggered deficiency windows")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, outdir / "Figure_6_threshold_windows.png")


def figure7_operational_threshold(outdir: Path) -> None:
    loads = np.linspace(0, 12, 13)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for L in loads:
        sol = M.simulate("chronic", cstress0_override=float(L), phenotype_overrides={"psi": "dep"})
        axes[0].plot(sol.t, sol.y[M.IDX["BDNF"]], lw=1.2, label=f"{L:g}")
    axes[0].axhline(60, ls="--", color="red", lw=1)
    axes[0].set_xlabel("time (h)"); axes[0].set_ylabel("BDNF (%)")
    axes[0].set_title("BDNF trajectories vs applied load")
    axes[0].grid(alpha=0.25)
    day7_bdnf=[]; day7_vitc=[]; day7_mg=[]
    load_grid = np.linspace(0, 12, 25)
    for L in load_grid:
        sol = M.simulate("chronic", cstress0_override=float(L), phenotype_overrides={"psi": "dep"})
        day7_bdnf.append(sol.y[M.IDX["BDNF"], -1]); day7_vitc.append(sol.y[M.IDX["VitC"], -1]); day7_mg.append(sol.y[M.IDX["Mg"], -1])
    axes[1].plot(load_grid, day7_bdnf, marker="o", label="BDNF (%)")
    axes[1].plot(load_grid, day7_vitc, marker="o", label="Vitamin C (µM)")
    axes[1].plot(load_grid, np.array(day7_mg)*100, marker="o", label="Mg x100")
    axes[1].axhline(60, ls="--", color="red", lw=1)
    # Interpolate threshold.
    bd = np.array(day7_bdnf)
    threshold_load = float(np.interp(60.0, bd[::-1], load_grid[::-1]))
    axes[1].axvline(threshold_load, ls=":", color="purple", lw=1.5)
    axes[1].set_xlabel("applied slow stress-load (µg/dL)"); axes[1].set_ylabel("day-7 value")
    axes[1].set_title(f"Operational BDNF boundary ≈ {threshold_load:.1f} µg/dL")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    fig.suptitle("Figure 7. Finite-horizon BDNF-HPA operational threshold", y=1.02)
    _save(fig, outdir / "Figure_7_bdnf_operational_threshold.png")


def generate_all(output_dir: str | Path = "outputs") -> None:
    out = Path(output_dir)
    figs = out / "figures"
    tables = out / "tables"
    _ensure(figs); _ensure(tables)
    export_tables(tables)
    figure2_day7_heatmap(figs)
    figure3_cortisol_backbone(figs)
    figure4_tier1(figs)
    figure5_tier2(figs)
    figure6_thresholds(figs)
    figure7_operational_threshold(figs)


def get_gui_spec():
    """Return the Tkinter task definition used by this script and the master launcher."""
    from tsbm_gui import TaskSpec

    def runner(output_dir: Path, options: dict) -> None:
        generate_all(output_dir)
        print(f"Generated Figures 2-7 and base tables in {output_dir}")

    return TaskSpec(
        title='TSBM Figure Generator',
        description='Generates manuscript Figures 2–7 and the base CSV tables from the ten-state TSBM model. Use the output folder selector, then press Run. Generated images appear in the gallery and can be saved as copies or opened in the basic editor.',
        runner=runner,
        default_output="outputs",
        run_button_text="Generate figures",
    )


if __name__ == "__main__":
    import argparse
    import sys

    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(description="Generate TSBM manuscript figures.")
        parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--output", default="outputs")
        args = parser.parse_args()
        generate_all(Path(args.output))
        print(f"Generated Figures 2-7 and base tables in {args.output}.")
    else:
        from tsbm_gui import launch_task_gui
        launch_task_gui(get_gui_spec())
