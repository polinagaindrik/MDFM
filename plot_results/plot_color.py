import os
import sys
sys.path.append(os.getcwd())
import fusion_model as fm
import matplotlib.pyplot as plt


def plot_color_values(*args, dir ='', val_name=''):
    fig, ax = plt.subplots(figsize=(6, 4.))
    fig, ax = fm.plotting.set_labels(fig, ax, 'Tag', val_name)
    for day, val, exp in zip(*args):
        ax.plot(day, val, label=exp, marker=fm.plotting.exp_mrkrs[exp[4:]], linestyle=fm.plotting.exp_lsts[exp[4:]],
                color=fm.plotting.exp_clrs[exp[:3]])
    ax.legend(fontsize=11, framealpha=0.1, loc='upper left', ncol=2, bbox_to_anchor=(1, 1.05))
    plt.savefig(dir+val_name, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = 500
    output_path = 'out/plot_color/'
    data_path = 'experiments/color/'
    # Define model, method, data used for model calibration 
    #username = ''
    #password = ''

    #fm.data.download_colordata_from_openbis(data_path, username, password=password)

    exps = [f'V{i:02d}' for i in range(1, 24)]
    df = fm.data.get_all_experiments_dataframe(fm.data.read_color, exps, data_path)
    df = fm.data.drop_column(df, ['PAT', 'V10_', 'CCD01-CCD02'])
    #df = df.filter(like='M1')
    exps2 = sorted(list(set([s.split('_')[0] for s in fm.data.drop_column(df, ['V15','V21', 'V22', 'V23']).columns])))

    # Do the plottings
    days_meas, days_meas_o, a, a_o, b, b_o, L, L_o = [[] for _ in range (8)]
    for exp in exps2:
        print(exp)
        df0 = df.filter(like=exp+'_').filter(like='M1').filter(like='_CLR')
        days_meas0 = [float(f.split('_')[3]) for f in df0.columns]
        a0  = [df0[f]['a*(D65)'] for f in df0.columns]
        b0  = [df0[f]['b*(D65)'] for f in df0.columns]
        L0  = [df0[f]['L*(D65)'] for f in df0.columns]
        days_meas0, a0, b0, L0 = zip(*sorted(zip(days_meas0, a0, b0, L0)))
        a.append(a0)
        b.append(b0)
        L.append(L0)
        days_meas.append(days_meas0)

    exps3 = sorted(list(set([s.split('_')[0] for s in fm.data.drop_column(df, ['V15','V21','V22','V23']).columns])))
    for exp in exps3:
        print(exp)
        df0_o = df.filter(like=exp+'_').filter(like='_o-').filter(like='M1')
        days_meas0_o = [float(f.split('_')[3]) for f in df0_o.columns]     
        a_o0 = [df0_o[f]['a*(D65)'] for f in df0_o.columns]
        b_o0 = [df0_o[f]['b*(D65)'] for f in df0_o.columns]
        L_o0 = [df0_o[f]['L*(D65)'] for f in df0_o.columns]
        days_meas0_o, a_o0, b_o0, L_o0 = zip(*sorted(zip(days_meas0_o, a_o0, b_o0, L_o0)))
        a_o.append(a_o0)
        b_o.append(b_o0)
        L_o.append(L_o0)
        days_meas_o.append(days_meas0_o)

    vals = [a, b, L]
    vals_o = [a_o, b_o, L_o]
    val_names = ['a', 'b', 'L']
    for v, vn in zip(vals, val_names):
        plot_color_values(days_meas, v, exps2, dir=output_path , val_name=vn)
    for v, vn in zip(vals_o, val_names):
        plot_color_values(days_meas_o, v, exps3, dir=output_path , val_name=vn+'_open')