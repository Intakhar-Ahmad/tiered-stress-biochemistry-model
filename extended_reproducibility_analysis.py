"""Extended reproducibility analyses for the revised TSBM preprint.

Reproduces threshold-crossing uncertainty, within-draw event order,
widened feedback-gain robustness, success-conditioned parameter trade-offs,
and the corresponding manuscript figures/tables.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, platform
import numpy as np
import scipy
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tsbm_model as M
from tsbm_rigor_analysis import KEY_PARAMETERS, latin_hypercube

DT = 0.5
T_END = 168.0
SEED_MODERATE = 20240101
SEED_WIDE = 20240102


def rk4_trajectory(name, p_overrides=None, dep_gain=0.60, ptsd_gain=0.15, dt=DT):
    p = M.parameters(p_overrides)
    ph = M.phenotype(name)
    y = M.initial_state(ph, p)
    times = np.arange(0.0, T_END + dt / 2, dt)
    ys = np.empty((len(times), len(y)), float)
    ys[0] = y
    for i in range(1, len(times)):
        t = times[i-1]
        h = times[i] - times[i-1]
        k1 = M.rhs(t, y, ph, p, dep_gain, ptsd_gain)
        k2 = M.rhs(t+h/2, y+h*k1/2, ph, p, dep_gain, ptsd_gain)
        k3 = M.rhs(t+h/2, y+h*k2/2, ph, p, dep_gain, ptsd_gain)
        k4 = M.rhs(t+h, y+h*k3, ph, p, dep_gain, ptsd_gain)
        y = y + h*(k1+2*k2+2*k3+k4)/6
        ys[i] = y
    return times, ys


def crossing_time(times, values, boundary, direction):
    cond = values < boundary if direction == 'below' else values > boundary
    idx = np.flatnonzero(cond)
    if not len(idx):
        return None
    i = int(idx[0])
    if i == 0:
        return float(times[0])
    x0, x1 = float(values[i-1]), float(values[i])
    t0, t1 = float(times[i-1]), float(times[i])
    if x1 == x0:
        return t1
    frac = (boundary-x0)/(x1-x0)
    return t0 + frac*(t1-t0)


def state_outputs(name, overrides=None, dep_gain=0.60, ptsd_gain=0.15):
    _, ys = rk4_trajectory(name, overrides, dep_gain, ptsd_gain)
    y = ys[-1]
    return {
        'VitC': float(y[M.IDX['VitC']]), 'Mg': float(y[M.IDX['Mg']]),
        'BDNF': float(y[M.IDX['BDNF']]), 'Nrf2': float(y[M.IDX['Nrf2']]),
        'INF': float(y[M.IDX['INF']]),
        'KYN_TRP': float(y[M.IDX['Kyn']]/max(y[M.IDX['Trp']],1e-12)),
    }


def criteria(dep, ptsd, chronic):
    bdnf = dep['BDNF'] < 70 and ptsd['BDNF'] > 85 and ptsd['BDNF']-dep['BDNF'] > 20
    multi = (bdnf and dep['Mg'] < .65 and dep['KYN_TRP'] > .08 and
             ptsd['Mg'] > .70 and ptsd['KYN_TRP'] < .08 and
             dep['BDNF'] < chronic['BDNF'] < ptsd['BDNF'])
    return bdnf, multi


def moderate_scan(n=400, seed=SEED_MODERATE):
    rng = np.random.default_rng(seed)
    lhs = latin_hypercube(n, len(KEY_PARAMETERS), rng)
    scales = 0.8 + 0.4*lhs
    rows=[]
    for i in range(n):
        ov={k:M.DEFAULT_PARAMETERS[k]*scales[i,j] for j,k in enumerate(KEY_PARAMETERS)}
        t,y=rk4_trajectory('depressed',ov)
        ratio=y[:,M.IDX['Kyn']]/np.maximum(y[:,M.IDX['Trp']],1e-12)
        crosses={
          'Mg':crossing_time(t,y[:,M.IDX['Mg']],.65,'below'),
          'BDNF':crossing_time(t,y[:,M.IDX['BDNF']],70,'below'),
          'KYN_TRP':crossing_time(t,ratio,.08,'above'),
          'VitC':crossing_time(t,y[:,M.IDX['VitC']],23,'below')}
        dep={k:float(v) for k,v in zip(['NE','VitC','Ald','Mg','BDNF','Nrf2','INF','Trp','Kyn','Cstress'],y[-1])}
        depout={'VitC':dep['VitC'],'Mg':dep['Mg'],'BDNF':dep['BDNF'],'Nrf2':dep['Nrf2'],'INF':dep['INF'],'KYN_TRP':dep['Kyn']/dep['Trp']}
        ptsd=state_outputs('ptsd',ov); chronic=state_outputs('chronic',ov)
        b,m=criteria(depout,ptsd,chronic)
        row={'draw':i+1,'bdnf_only_ok':b,'multi_output_ok':m,**{f'scale_{k}':float(scales[i,j]) for j,k in enumerate(KEY_PARAMETERS)}}
        row.update({f'cross_{k}_h':v for k,v in crosses.items()})
        for prefix,vals in [('dep',depout),('ptsd',ptsd),('chronic',chronic)]:
            row.update({f'{prefix}_{k}':float(v) for k,v in vals.items()})
        valid=[crosses[k] for k in ['Mg','BDNF','KYN_TRP','VitC']]
        row['all_four_crossed']=all(v is not None for v in valid)
        row['nominal_order']=row['all_four_crossed'] and valid==sorted(valid)
        rows.append(row)
    return rows


def wide_scan(n=400, seed=SEED_WIDE):
    rng=np.random.default_rng(seed)
    lhs=latin_hypercube(n,len(KEY_PARAMETERS)+2,rng)
    scales=.8+.4*lhs[:,:len(KEY_PARAMETERS)]
    dep_gains=.60*10**(-1+2*lhs[:,-2])
    ptsd_gains=.15*10**(-1+2*lhs[:,-1])
    rows=[]
    for i in range(n):
        ov={k:M.DEFAULT_PARAMETERS[k]*scales[i,j] for j,k in enumerate(KEY_PARAMETERS)}
        dep=state_outputs('depressed',ov,float(dep_gains[i]),float(ptsd_gains[i]))
        ptsd=state_outputs('ptsd',ov,float(dep_gains[i]),float(ptsd_gains[i]))
        chronic=state_outputs('chronic',ov,float(dep_gains[i]),float(ptsd_gains[i]))
        b,m=criteria(dep,ptsd,chronic)
        rows.append({'draw':i+1,'dep_gain':float(dep_gains[i]),'ptsd_gain':float(ptsd_gains[i]),'bdnf_only_ok':b,'multi_output_ok':m})
    return rows


def summary(rows):
    out={}
    for key in ['Mg','BDNF','KYN_TRP','VitC']:
        vals=[r[f'cross_{key}_h'] for r in rows if r[f'cross_{key}_h'] is not None]
        out[key]={'crossing_fraction':len(vals)/len(rows),'median':float(np.median(vals)),'q2_5':float(np.percentile(vals,2.5)),'q97_5':float(np.percentile(vals,97.5))}
    out['all_four_crossed_n']=sum(r['all_four_crossed'] for r in rows)
    out['nominal_order_n']=sum(r['nominal_order'] for r in rows)
    out['nominal_order_fraction_all']=out['nominal_order_n']/len(rows)
    out['nominal_order_fraction_complete']=out['nominal_order_n']/out['all_four_crossed_n']
    out['bdnf_only_fraction']=sum(r['bdnf_only_ok'] for r in rows)/len(rows)
    out['multi_output_fraction']=sum(r['multi_output_ok'] for r in rows)/len(rows)
    return out


def tradeoff(rows):
    success=[r for r in rows if r['multi_output_ok']]
    matrix=np.array([[r[f'scale_{k}'] for k in KEY_PARAMETERS] for r in success])
    corr,_=spearmanr(matrix,axis=0)
    pairs=[]
    for i in range(len(KEY_PARAMETERS)):
        for j in range(i+1,len(KEY_PARAMETERS)):
            pairs.append({'parameter_1':KEY_PARAMETERS[i],'parameter_2':KEY_PARAMETERS[j],'spearman_rho':float(corr[i,j])})
    pairs.sort(key=lambda x:abs(x['spearman_rho']),reverse=True)
    return {'n_success':len(success),'max_abs_rho':abs(pairs[0]['spearman_rho']) if pairs else None,'pairs':pairs}


def save_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def make_figures(out,mod,wide):
    labels=['Magnesium','BDNF','KYN/TRP','Vitamin C']
    keys=['Mg','BDNF','KYN_TRP','VitC']
    med=[mod[k]['median'] for k in keys]
    lo=[mod[k]['median']-mod[k]['q2_5'] for k in keys]
    hi=[mod[k]['q97_5']-mod[k]['median'] for k in keys]
    fig,ax=plt.subplots(figsize=(7.8,4.5))
    y=np.arange(4)
    ax.errorbar(med,y,xerr=[lo,hi],fmt='o',capsize=4)
    for i,k in enumerate(keys): ax.text(mod[k]['q97_5']+3,i,f"{100*mod[k]['crossing_fraction']:.1f}%",va='center',fontsize=9)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.axvline(24,ls=':',lw=1); ax.axvline(72,ls=':',lw=1)
    ax.set_xlabel('First crossing time (h)'); ax.set_title('Depression-like threshold-crossing uncertainty')
    ax.grid(axis='x',alpha=.25); fig.tight_layout(); fig.savefig(out/'Figure_6_threshold_uncertainty.png',dpi=300,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    x=np.arange(2); width=.34
    moderate=[100*mod['bdnf_only_fraction'],100*mod['multi_output_fraction']]
    widened=[100*wide['bdnf_only_fraction'],100*wide['multi_output_fraction']]
    ax.bar(x-width/2,moderate,width,label='All 14 parameters +/-20%')
    ax.bar(x+width/2,widened,width,label='Plus feedback gains 0.1-10x')
    ax.set_xticks(x,['BDNF-only criterion','Multi-output criterion']); ax.set_ylim(0,105); ax.set_ylabel('Draws retaining criterion (%)')
    ax.legend(frameon=False); ax.set_title('Robustness to moderate and widened feedback-gain uncertainty')
    for bars in ax.containers: ax.bar_label(bars,fmt='%.0f%%',padding=3)
    fig.tight_layout(); fig.savefig(out/'Figure_9_robustness.png',dpi=300,bbox_inches='tight'); plt.close(fig)


def run(output_dir='extended_outputs',n=400):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    moderate=moderate_scan(n); wide=wide_scan(n)
    modsum=summary(moderate)
    widesum={'n':n,'seed':SEED_WIDE,'bdnf_only_fraction':sum(r['bdnf_only_ok'] for r in wide)/n,'multi_output_fraction':sum(r['multi_output_ok'] for r in wide)/n}
    tr=tradeoff(moderate)
    save_csv(out/'threshold_uncertainty_draws.csv',moderate)
    save_csv(out/'widened_feedback_draws.csv',wide)
    save_csv(out/'success_conditioned_spearman_pairs.csv',tr['pairs'])
    manifest={'n':n,'moderate_seed':SEED_MODERATE,'widened_seed':SEED_WIDE,'fixed_step_h':DT,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'moderate_summary':modsum,'widened_summary':widesum,'tradeoff':{'n_success':tr['n_success'],'max_abs_rho':tr['max_abs_rho']}}
    with open(out/'extended_run_manifest.json','w') as f: json.dump(manifest,f,indent=2)
    make_figures(out,modsum,widesum)
    print(json.dumps(manifest,indent=2))
    return manifest


def get_gui_spec():
    """Return the Tkinter task definition used by this script and the master launcher."""
    from tsbm_gui import OptionSpec, TaskSpec

    def runner(output_dir: Path, options: dict) -> None:
        n = int(options["n"])
        if n < 2:
            raise ValueError("The number of draws must be at least 2.")
        run(output_dir, n=n)

    return TaskSpec(
        title="TSBM Extended Reproducibility Analysis",
        description=(
            "Runs threshold-crossing uncertainty, within-draw event ordering, widened feedback-gain "
            "robustness, and success-conditioned parameter trade-off analyses. It writes detailed CSV/JSON "
            "outputs and generates Figures 6 and 9 for review."
        ),
        runner=runner,
        default_output="extended_outputs",
        options=(
            OptionSpec(
                key="n", label="Latin-hypercube draws", kind="int", default=400,
                help_text="Use 20-50 for a quick test or 400 for the manuscript run.",
            ),
        ),
        run_button_text="Run extended analyses",
        warning="A 400-draw run may take several minutes. The GUI remains responsive while it runs.",
    )


if __name__ == "__main__":
    import argparse
    import sys

    if "--cli" in sys.argv:
        parser = argparse.ArgumentParser(description="Run extended TSBM reproducibility analyses.")
        parser.add_argument("--cli", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--output", default="extended_outputs")
        parser.add_argument("--n", type=int, default=400)
        args = parser.parse_args()
        run(args.output, args.n)
    else:
        from tsbm_gui import launch_task_gui
        launch_task_gui(get_gui_spec())
