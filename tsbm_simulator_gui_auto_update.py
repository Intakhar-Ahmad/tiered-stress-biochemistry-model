"""
TSBM Interactive Simulator (Tkinter GUI)
=======================================
A desktop application for exploring the ten-equation Tiered Stress Biochemistry
Model (TSBM).

Run:
    python tsbm_simulator_gui.py

Features
  - select any combination of five phenotypes
  - plot state variables, KYN/TRP ratio, or total cortisol
  - adjust key model parameters interactively
  - export currently plotted traces to CSV
  - save the current plot as PNG
"""
from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tsbm_model as M

COLORS = {
    "normal": "#2e8b57",
    "acute": "#e0a458",
    "chronic": "#3a5a99",
    "depressed": "#c1440e",
    "ptsd": "#6a4c93",
}
PLOTTABLE = ["NE", "VitC", "Mg", "Ald", "BDNF", "Nrf2", "INF", "Trp", "Kyn", "KYN/TRP", "Total cortisol"]
YLABEL = {
    "NE": "Noradrenergic drive (relative)",
    "VitC": "Vitamin C (µM)",
    "Mg": "Magnesium (mmol/L; proxy)",
    "Ald": "Aldosterone (a.u.)",
    "BDNF": "BDNF (%)",
    "Nrf2": "Nrf2 (%)",
    "INF": "Inflammation (a.u.)",
    "Trp": "Tryptophan (µM)",
    "Kyn": "Kynurenine (µM)",
    "KYN/TRP": "KYN/TRP ratio",
    "Total cortisol": "Cortisol (µg/dL)",
}


def series(sol, key: str) -> np.ndarray:
    if key == "KYN/TRP":
        return M.kyntrp(sol)
    if key == "Total cortisol":
        return sol.Ctot
    return sol.y[M.IDX[key]]


class TSBMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSBM Interactive Simulator")
        self.geometry("1180x720")
        self.last_results: dict[str, object] = {}
        self.last_variable = "BDNF"

        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="TSBM Simulator", font=("Helvetica", 14, "bold")).pack(anchor="w")
        ttk.Label(left, text="Ten-equation stress biochemistry model", foreground="#555").pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Phenotypes", font=("Helvetica", 10, "bold")).pack(anchor="w")
        self.pheno_vars = {}
        for ph in M.PHENOTYPE_ORDER:
            var = tk.BooleanVar(value=ph in ("normal", "depressed", "ptsd"))
            self.pheno_vars[ph] = var
            ttk.Checkbutton(left, text=ph.capitalize(), variable=var).pack(anchor="w")

        ttk.Label(left, text="Variable to plot", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.var_choice = tk.StringVar(value="BDNF")
        self.variable_box = ttk.Combobox(
            left,
            textvariable=self.var_choice,
            values=PLOTTABLE,
            state="readonly",
            width=22,
        )
        self.variable_box.pack(anchor="w")
        # Redraw immediately when the user selects a different plotted variable.
        self.variable_box.bind("<<ComboboxSelected>>", self._on_variable_change)

        self.sliders = {}
        self._slider(left, "Depression feedback gain", "dep_gain", 0.0, 1.5, 0.60)
        self._slider(left, "PTSD feedback gain", "ptsd_gain", 0.0, 0.5, 0.15)
        self._slider(left, "Allostatic accumulation kacc", "kacc", 0.0, 0.06, 0.028)
        self._slider(left, "BDNF silencing ksil_bdnf", "ksil_bdnf", 0.0, 0.25, 0.100)
        self._slider(left, "TH chronic induction k_thind", "k_thind", 0.0, 0.06, 0.020)
        self._slider(left, "Mg wasting kw_mg", "kw_mg", 0.0, 0.08, 0.040)
        self._slider(left, "Mg physiologic floor", "Mg_floor", 0.30, 0.70, 0.55)
        self._slider(left, "Simulation length (h)", "t_end", 24, 336, 168)

        ttk.Button(left, text="Run simulation", command=self.run).pack(anchor="w", pady=(12, 4), fill=tk.X)
        ttk.Button(left, text="Export plotted traces CSV", command=self.export_csv).pack(anchor="w", pady=2, fill=tk.X)
        ttk.Button(left, text="Save current plot PNG", command=self.save_plot).pack(anchor="w", pady=2, fill=tk.X)

        ttk.Label(left, text="Day-7 values and crossings", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(10, 0))
        self.results_box = tk.Text(left, width=40, height=15, font=("Courier", 8))
        self.results_box.pack(anchor="w", fill=tk.Y)

        self.fig = Figure(figsize=(7.8, 6.1), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, right)
        self.run()

    def _slider(self, parent, label: str, key: str, min_val: float, max_val: float, init: float) -> None:
        ttk.Label(parent, text=label, font=("Helvetica", 9)).pack(anchor="w", pady=(8, 0))
        var = tk.DoubleVar(value=init)
        slider = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, length=240)
        slider.pack(anchor="w")
        value_label = ttk.Label(parent, text=f"{init:.4g}", foreground="#337")
        value_label.pack(anchor="w")
        var.trace_add("write", lambda *_: value_label.config(text=f"{var.get():.4g}"))
        self.sliders[key] = var

    def _on_variable_change(self, _event=None) -> None:
        """Refresh the plot immediately after a new variable is selected."""
        self.run()

    def _parameter_overrides(self) -> dict:
        return {
            "kacc": self.sliders["kacc"].get(),
            "ksil_bdnf": self.sliders["ksil_bdnf"].get(),
            "k_thind": self.sliders["k_thind"].get(),
            "kw_mg": self.sliders["kw_mg"].get(),
            "Mg_floor": self.sliders["Mg_floor"].get(),
        }

    def run(self) -> None:
        selected = [ph for ph, var in self.pheno_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("No phenotype selected", "Select at least one phenotype.")
            return
        key = self.var_choice.get()
        t_end = float(self.sliders["t_end"].get())
        dep_gain = self.sliders["dep_gain"].get()
        ptsd_gain = self.sliders["ptsd_gain"].get()
        overrides = self._parameter_overrides()

        self.ax.clear()
        report_lines = []
        results = {}
        for ph in selected:
            sol = M.simulate(ph, t_end=t_end, n=1500, parameter_overrides=overrides, dep_gain=dep_gain, ptsd_gain=ptsd_gain)
            arr = series(sol, key)
            results[ph] = {"t": sol.t, "series": arr, "solution": sol}
            self.ax.plot(sol.t, arr, color=COLORS[ph], lw=1.8, label=ph)

            kt = M.kyntrp(sol)
            report_lines.extend([
                f"[{ph}]",
                f"  VitC {sol.y[M.IDX['VitC'], -1]:6.1f} µM   Mg {sol.y[M.IDX['Mg'], -1]:.3f}",
                f"  BDNF {sol.y[M.IDX['BDNF'], -1]:6.1f}%   Nrf2 {sol.y[M.IDX['Nrf2'], -1]:.1f}%",
                f"  INF  {sol.y[M.IDX['INF'], -1]:6.2f}     KT {kt[-1]:.3f}",
                f"  Mg<0.65: {self._fmt(M.cross_time(sol, sol.y[M.IDX['Mg']], 0.65))}",
                f"  BDNF<70: {self._fmt(M.cross_time(sol, sol.y[M.IDX['BDNF']], 70.0))}",
                f"  KT>0.08: {self._fmt(M.cross_time(sol, kt, 0.08, below=False))}",
                "",
            ])

        if key in ("Mg", "BDNF", "VitC", "KYN/TRP"):
            threshold = {"Mg": 0.65, "BDNF": 70.0, "VitC": 23.0, "KYN/TRP": 0.08}[key]
            self.ax.axhline(threshold, color="red", ls="--", lw=1, alpha=0.7)
        self.ax.set_xlabel("time (h)")
        self.ax.set_ylabel(YLABEL[key])
        self.ax.set_title(f"TSBM: {key} by phenotype")
        self.ax.grid(alpha=0.25)
        self.ax.legend(fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

        self.last_results = results
        self.last_variable = key
        self.results_box.delete("1.0", tk.END)
        self.results_box.insert(tk.END, "\n".join(report_lines))

    @staticmethod
    def _fmt(value) -> str:
        return f"{value:.1f} h" if value is not None else "not crossed"

    def export_csv(self) -> None:
        if not self.last_results:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="tsbm_traces.csv")
        if not path:
            return
        # Union of time grids is not needed because all selected simulations use the same grid.
        first = next(iter(self.last_results.values()))
        t = first["t"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_h"] + [f"{ph}_{self.last_variable}" for ph in self.last_results])
            for i in range(len(t)):
                writer.writerow([t[i]] + [data["series"][i] for data in self.last_results.values()])
        messagebox.showinfo("Export complete", f"Saved {Path(path).name}")

    def save_plot(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile="tsbm_plot.png")
        if not path:
            return
        self.fig.savefig(path, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Plot saved", f"Saved {Path(path).name}")


if __name__ == "__main__":
    TSBMApp().mainloop()
