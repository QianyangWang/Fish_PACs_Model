import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['mathtext.default'] = 'it'

namedic = {
    "log(Km,N)": r"log($\mathit{K}_{\mathrm{m,N}}$)",
    "kqw": r"$\mathit{k}_{\mathrm{qw}}$",
    "nqw": r"$\mathit{n}_{\mathrm{qw}}$",
    "uf": r"$\mathit{u}_{\mathrm{f}}$",
    "log(BAF)": r"log($\mathit{BAF}^{\prime}$)",
    "rsed": r"$\mathit{r}_{\mathrm{sed}}$",
    "log(BSAF)": r"log($\mathit{BSAF}$)",
    "af": r"$\mathit{a}_{\mathrm{f}}$",
    "e_doc": r"$\mathit{e}_{\mathrm{doc}}$",
    "e_dis": r"$\mathit{e}_{\mathrm{dis}}$",
    "kql": r"$\mathit{k}_{\mathrm{ql}}$",
    "e_par": r"$\mathit{e}_{\mathrm{p}}$",
    "s": r"$\mathit{s}$",
    "log(cKlw)": r"log($\mathit{cK}_{\mathrm{lw}}$)"
}

slscres = pd.read_excel(r"Results\SensitivitySLSC_PC4.xlsx")
lkcbres = pd.read_excel(r"Results\SensitivityLKCH_PC4.xlsx")
slscres['Group'] = 'SLSC'
lkcbres['Group'] = 'LKCB'

colors = {'SLSC': 'forestgreen', 'LKCB': 'darkorchid'}

all_cv = pd.concat([slscres['CV'], lkcbres['CV']])
cv_min, cv_max = all_cv.min(), all_cv.max()
size_min, size_max = 60, 600


def cv_to_size(cv):
    if cv_max > cv_min:
        norm = (cv - cv_min) / (cv_max - cv_min)
        norm = np.clip(norm, 0, 1)
    else:
        norm = np.zeros_like(cv)
    return size_min + norm * (size_max - size_min)


low_r = np.floor(cv_min * 10) / 10
high_r = np.ceil(cv_max * 10) / 10
mid_r = round((low_r + high_r) / 2, 1)
cv_examples = sorted(set([low_r, mid_r, high_r]))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharex=False)

for ax, (group, gdf) in zip(axes, [('SLSC', slscres), ('LKCB', lkcbres)]):
    gdf = gdf.copy()
    gdf = gdf.sort_values('Mean', ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(gdf))
    labels = [namedic.get(p, p) for p in gdf['Parameter']]

    mean_val_line = gdf['Mean'].mean()
    ax.axvline(mean_val_line, color=colors[group], linewidth=1.5,
               linestyle=":", zorder=1)

    ax.hlines(y_pos, 0, gdf['Mean'],
              color=colors[group], alpha=0.5, linewidth=2.2, zorder=2)

    cv_mean_group = gdf['CV'].mean()
    edge_colors = np.where(gdf['CV'] > cv_mean_group, 'red', 'black')
    edge_widths = np.where(gdf['CV'] > cv_mean_group, 2.0, 1.3)

    ax.scatter(gdf['Mean'], y_pos,
               s=cv_to_size(gdf['CV']),
               c=colors[group],
               alpha=0.9,
               edgecolors=edge_colors,
               linewidth=edge_widths,
               zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Mean PAWN Index', fontsize=13, labelpad=5)

    ax.grid(True, axis='x', alpha=0.2, linestyle='--', linewidth=0.5, zorder=0)
    ax.set_facecolor('#F8F8F8')

    for spine_name in ['top', 'right']:
        ax.spines[spine_name].set_linewidth(1.2)
    for spine_name in ['bottom', 'left']:
        ax.spines[spine_name].set_linewidth(2)

    ax.set_xlim(0, max(gdf['Mean'].max() * 1.12, 0.1))
    ax.tick_params(labelsize=11)

    size_handles = [
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                              markeredgecolor='black',
                              markersize=np.sqrt(cv_to_size(cv)) * 0.7,
                              label=f'{cv}')
                       for cv in cv_examples
                   ]

    edge_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=14, markeredgewidth=2.0,
               label='CV \u2264 group mean'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='red', markersize=14, markeredgewidth=2.8,
               label='CV > group mean'),
    ]

    leg = ax.legend(handles=size_handles + edge_handles,
                    loc='lower right',
                    fontsize=10,
                    title='Bubble Properties: CV',
                    title_fontsize=10,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='gray',
                    facecolor='#F8F8F8',
                    borderpad=0.9,
                    labelspacing=0.8,
                    handletextpad=0.9)
    leg._legend_box.align = "left"
    leg.get_title().set_ha('left')

plt.tight_layout()

plt.savefig(r"Results\SensitivityPC4.jpg",dpi=600, bbox_inches='tight')
plt.show()