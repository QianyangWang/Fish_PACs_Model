import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from scipy.stats import pearsonr

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


idx_x, idx_y = 8, -1
labels = [r"log($\mathit{cK}_{\mathrm{lw}}$)", r"log($\mathit{K}_{\mathrm{m,N}}$)"]

allparam1 = np.load(r"Results\LKCH\ParamsLKCH_PC4.npy")
allparam2 = np.load(r"Results\SLSC\ParamsSLSC_PC4.npy")
behav_samp1 = np.load(r"Results\LKCH\Behavioral\BehaviouralParams.npy")
behav_samp2 = np.load(r"Results\SLSC\Behavioral\BehaviouralParams.npy")
behav_w1 = np.load(r"Results\LKCH\Behavioral\BehaviouralWeights.npy")
behav_w2 = np.load(r"Results\SLSC\Behavioral\BehaviouralWeights.npy")

x_lk, y_lk, w_lk = behav_samp1[:, idx_x], behav_samp1[:, idx_y], behav_w1
x_sl, y_sl, w_sl = behav_samp2[:, idx_x], behav_samp2[:, idx_y], behav_w2

x_lk_prior, y_lk_prior = allparam1[:, idx_x], allparam1[:, idx_y]
x_sl_prior, y_sl_prior = allparam2[:, idx_x], allparam2[:, idx_y]

lb_x = min(x_lk_prior.min(), x_sl_prior.min())
ub_x = max(x_lk_prior.max(), x_sl_prior.max())
lb_y = min(y_lk_prior.min(), y_sl_prior.min())
ub_y = max(y_lk_prior.max(), y_sl_prior.max())

lb_x = float(lb_x) + 0.0
ub_x = float(ub_x) + 0.0
lb_y = float(lb_y) + 0.0
ub_y = float(ub_y) + 0.0

prior_density_x = 1.0 / (ub_x - lb_x)
prior_density_y = 1.0 / (ub_y - lb_y)

colors = {'LKCH': 'darkorchid', 'SLSC': 'forestgreen'}
prior_color = 'gray'
median_line_color = 'black'


FS_LABEL   = 42
FS_TICK    = 36
FS_LEGEND  = 22
FS_MEDIAN  = 31
FS_DENS    = 39
FS_CORR_H  = 26
FS_CORR_R  = 24
FS_CORR_P  = 22


LW_KDE      = 4.2
LW_PRIOR    = 3.3
LW_MEDIAN   = 4.5
LW_SPINE    = 1.8
SZ_TICKLEN  = 21
LW_TICKLEN  = 2.7

PRIOR_LS = ':'
MEDIAN_LS = '--'


def weighted_quantile(vals, w, q):
    order = np.argsort(vals)
    v, ww = vals[order], w[order]
    cw = np.cumsum(ww)
    return v[np.searchsorted(cw, q * cw[-1])]

def weighted_kde(vals, w, n_points=300, grid=None):
    w = w / w.sum()
    if grid is None:
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin if vmax > vmin else 1.0
        grid = np.linspace(vmin - 0.1 * span, vmax + 0.1 * span, n_points)
    bw = np.std(vals) * (len(vals) ** (-0.2))
    density = np.zeros_like(grid)
    for i in range(len(vals)):
        density += w[i] * np.exp(-0.5 * ((grid - vals[i]) / bw) ** 2)
    density /= (bw * np.sqrt(2 * np.pi))
    density /= np.trapz(density, grid)
    return grid, density

def subsample(x, y, w=None, n_max=20000, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(x), n_max)
    if w is None:
        sel = rng.choice(len(x), size=n, replace=False)
    else:
        p = w / w.sum()
        sel = rng.choice(len(x), size=n, replace=True, p=p)
    return x[sel], y[sel]

def snap_tick_to_value(ax, axis, target):
    ticks = list(ax.get_xticks() if axis == 'x' else ax.get_yticks())
    if len(ticks) == 0:
        return
    idx = int(np.argmin(np.abs(np.array(ticks) - target)))
    ticks[idx] = float(target) + 0.0
    if axis == 'x':
        ax.set_xticks(ticks)
    else:
        ax.set_yticks(ticks)

def fmt_tick(x, pos=None):
    xr = round(x, 1)
    if abs(xr) < 1e-10:
        return '0.0'
    return f'{xr:.1f}'


fig = plt.figure(figsize=(12.5, 12.5))
gs = GridSpec(2, 2, figure=fig,
              width_ratios=[5, 1.6], height_ratios=[1.6, 5],
              wspace=0.0, hspace=0.0)

ax_scatter = fig.add_subplot(gs[1, 0])
ax_histx   = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_histy   = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
ax_corr    = fig.add_subplot(gs[0, 1])


xs_lk_p, ys_lk_p = subsample(x_lk_prior, y_lk_prior, None, n_max=8000, seed=1)
xs_sl_p, ys_sl_p = subsample(x_sl_prior, y_sl_prior, None, n_max=8000, seed=2)

ax_scatter.scatter(xs_sl_p, ys_sl_p, s=8, alpha=0.10,
                   color=prior_color, edgecolors='none', rasterized=True)
ax_scatter.scatter(xs_lk_p, ys_lk_p, s=8, alpha=0.10,
                   color=prior_color, edgecolors='none', rasterized=True)

xs_lk, ys_lk = subsample(x_lk, y_lk, w_lk, seed=42)
xs_sl, ys_sl = subsample(x_sl, y_sl, w_sl, seed=43)

ax_scatter.scatter(xs_sl, ys_sl, s=19, alpha=0.35,
                   color=colors['SLSC'], edgecolors='none', rasterized=True)
ax_scatter.scatter(xs_lk, ys_lk, s=19, alpha=0.35,
                   color=colors['LKCH'], edgecolors='none', rasterized=True)

scatter_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['SLSC'],
           markersize=16, alpha=0.8, label='SLSC Behavioral'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['LKCH'],
           markersize=16, alpha=0.8, label='LKCH Behavioral'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=prior_color,
           markersize=16, alpha=0.5, label='Prior'),
    Line2D([0], [0], color=median_line_color, lw=LW_MEDIAN,
           linestyle=MEDIAN_LS, label='Median'),
]
ax_scatter.legend(handles=scatter_handles, loc='lower right',
                  fontsize=FS_LEGEND, frameon=True, framealpha=0.9,
                  edgecolor='gray', facecolor='white',
                  handletextpad=0.6, borderpad=0.6,
                  labelspacing=0.5)


grid_x = np.linspace(lb_x, ub_x, 400)
grid_y = np.linspace(lb_y, ub_y, 400)

_, pdfx_lk = weighted_kde(x_lk, w_lk, grid=grid_x)
_, pdfy_lk = weighted_kde(y_lk, w_lk, grid=grid_y)
_, pdfx_sl = weighted_kde(x_sl, w_sl, grid=grid_x)
_, pdfy_sl = weighted_kde(y_sl, w_sl, grid=grid_y)

ax_histx.hlines(prior_density_x, lb_x, ub_x,
                color=prior_color, linestyle=PRIOR_LS,
                linewidth=LW_PRIOR, alpha=0.9, zorder=1)
ax_histx.plot(grid_x, pdfx_sl, color=colors['SLSC'], linewidth=LW_KDE, zorder=3)
ax_histx.plot(grid_x, pdfx_lk, color=colors['LKCH'], linewidth=LW_KDE, zorder=3)

ax_histy.vlines(prior_density_y, lb_y, ub_y,
                color=prior_color, linestyle=PRIOR_LS,
                linewidth=LW_PRIOR, alpha=0.9, zorder=1)
ax_histy.plot(pdfy_sl, grid_y, color=colors['SLSC'], linewidth=LW_KDE, zorder=3)
ax_histy.plot(pdfy_lk, grid_y, color=colors['LKCH'], linewidth=LW_KDE, zorder=3)

kde_handles = [
    Line2D([0], [0], color=colors['SLSC'], lw=LW_KDE,
           label='SLSC'),
    Line2D([0], [0], color=colors['LKCH'], lw=LW_KDE,
           label='LKCH'),
    Line2D([0], [0], color=prior_color, lw=LW_PRIOR, linestyle=PRIOR_LS,
           label='Prior'),
    Line2D([0], [0], color=median_line_color, lw=LW_MEDIAN,
           linestyle=MEDIAN_LS, label='Median'),
]
ax_histx.legend(handles=kde_handles, loc='lower right',
                fontsize=FS_LEGEND, frameon=True, framealpha=0.9,
                edgecolor='gray', facecolor='white',
                handlelength=2.0, labelspacing=0.5,
                borderpad=0.6)


qx50_lk = weighted_quantile(x_lk, w_lk, 0.5)
qy50_lk = weighted_quantile(y_lk, w_lk, 0.5)
qx50_sl = weighted_quantile(x_sl, w_sl, 0.5)
qy50_sl = weighted_quantile(y_sl, w_sl, 0.5)

ax_histx.axvline(qx50_lk, color=colors['LKCH'], linewidth=LW_MEDIAN,
                 linestyle=MEDIAN_LS, alpha=0.85)
ax_histx.axvline(qx50_sl, color=colors['SLSC'], linewidth=LW_MEDIAN,
                 linestyle=MEDIAN_LS, alpha=0.85)

peak_top = max(pdfx_lk.max(), pdfx_sl.max())
ymax_x = peak_top * 1.15
ax_histx.set_ylim(0, ymax_x)


top_y = ymax_x * 0.90
small_top = min(qx50_lk, qx50_sl)
large_top = max(qx50_lk, qx50_sl)

if qx50_sl <= qx50_lk:
    color_small_top = colors['SLSC']
    color_large_top = colors['LKCH']
else:
    color_small_top = colors['LKCH']
    color_large_top = colors['SLSC']

ax_histx.annotate(f"{small_top:.2f}", xy=(small_top, top_y),
                  xytext=(-8, 0), textcoords='offset points',
                  ha='right', va='center', color=color_small_top,
                  fontsize=FS_MEDIAN, fontweight='bold', zorder=7)
ax_histx.annotate(f"{large_top:.2f}", xy=(large_top, top_y),
                  xytext=(8, 0), textcoords='offset points',
                  ha='left', va='center', color=color_large_top,
                  fontsize=FS_MEDIAN, fontweight='bold', zorder=7)

ax_histy.axhline(qy50_lk, color=colors['LKCH'], linewidth=LW_MEDIAN,
                 linestyle=MEDIAN_LS, alpha=0.85)
ax_histy.axhline(qy50_sl, color=colors['SLSC'], linewidth=LW_MEDIAN,
                 linestyle=MEDIAN_LS, alpha=0.85)

peak_right = max(pdfy_lk.max(), pdfy_sl.max())
xmax_y = peak_right * 1.15
ax_histy.set_xlim(0, xmax_y)


right_x = xmax_y * 0.95
small_right = min(qy50_lk, qy50_sl)
large_right = max(qy50_lk, qy50_sl)

if qy50_sl <= qy50_lk:
    color_small_right = colors['SLSC']
    color_large_right = colors['LKCH']
else:
    color_small_right = colors['LKCH']
    color_large_right = colors['SLSC']

ax_histy.annotate(f"{small_right:.2f}", xy=(right_x, small_right),
                  xytext=(0, -8), textcoords='offset points',
                  ha='center', va='top', color=color_small_right,
                  fontsize=FS_MEDIAN, fontweight='bold', zorder=7)
ax_histy.annotate(f"{large_right:.2f}", xy=(right_x, large_right),
                  xytext=(0, 8), textcoords='offset points',
                  ha='center', va='bottom', color=color_large_right,
                  fontsize=FS_MEDIAN, fontweight='bold', zorder=7)

ax_scatter.axvline(qx50_lk, color=colors['LKCH'], linewidth=LW_MEDIAN,
                   linestyle=MEDIAN_LS, alpha=0.5)
ax_scatter.axvline(qx50_sl, color=colors['SLSC'], linewidth=LW_MEDIAN,
                   linestyle=MEDIAN_LS, alpha=0.5)
ax_scatter.axhline(qy50_lk, color=colors['LKCH'], linewidth=LW_MEDIAN,
                   linestyle=MEDIAN_LS, alpha=0.5)
ax_scatter.axhline(qy50_sl, color=colors['SLSC'], linewidth=LW_MEDIAN,
                   linestyle=MEDIAN_LS, alpha=0.5)


r_lk, p_lk = pearsonr(x_lk, y_lk)
r_sl, p_sl = pearsonr(x_sl, y_sl)

def fmt_p(p):
    if p < 0.001:
        return "p < 0.001"
    else:
        return f"p = {p:.3f}"

CORR_X = 0.28

ax_corr.set_xticks([])
ax_corr.set_yticks([])

ax_corr.text(CORR_X, 0.95, "SLSC",
             ha='left', va='top', transform=ax_corr.transAxes,
             fontsize=FS_CORR_H, color=colors['SLSC'], fontweight='bold')
ax_corr.text(CORR_X, 0.79, f"r = {r_sl:.2f}",
             ha='left', va='top', transform=ax_corr.transAxes,
             fontsize=FS_CORR_R, color=colors['SLSC'])
ax_corr.text(CORR_X, 0.65, fmt_p(p_sl),
             ha='left', va='top', transform=ax_corr.transAxes,
             fontsize=FS_CORR_P, color=colors['SLSC'])

ax_corr.text(CORR_X, 0.45, "LKCH",
             ha='left', va='top', transform=ax_corr.transAxes,
             fontsize=FS_CORR_H, color=colors['LKCH'], fontweight='bold')
ax_corr.text(CORR_X, 0.29, f"r = {r_lk:.2f}",
             ha='left', va='top', transform=ax_corr.transAxes,
             fontsize=FS_CORR_R, color=colors['LKCH'])
ax_corr.text(CORR_X, 0.15, fmt_p(p_lk),
             ha='left', va='top', transform=ax_corr.transAxes,
             fontsize=FS_CORR_P, color=colors['LKCH'])


ax_scatter.set_xlim(lb_x, ub_x)
ax_scatter.set_ylim(lb_y, ub_y)
ax_scatter.tick_params(axis='both', labelsize=FS_TICK)

ax_histx.set_xlim(lb_x, ub_x)
ax_histx.tick_params(axis='x', labelbottom=False)
ax_histx.tick_params(axis='y', labelsize=FS_TICK)

ax_histy.set_ylim(lb_y, ub_y)
ax_histy.tick_params(axis='y', labelleft=False)
ax_histy.tick_params(axis='x', labelsize=FS_TICK)

for ax in [ax_scatter, ax_histx, ax_histy, ax_corr]:
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(LW_SPINE)
        s.set_color('black')


snap_tick_to_value(ax_scatter, 'x', ub_x)
snap_tick_to_value(ax_histy,   'x', 0.0)
snap_tick_to_value(ax_histx,   'y', 0.0)
snap_tick_to_value(ax_scatter, 'y', ub_y)


def force_tick_labels(ax, axis):
    if axis == 'x':
        vals = ax.get_xticks()
        strs = [fmt_tick(v) for v in vals]
        ax.set_xticks(vals)
        ax.set_xticklabels(strs)
    else:
        vals = ax.get_yticks()
        strs = [fmt_tick(v) for v in vals]
        ax.set_yticks(vals)
        ax.set_yticklabels(strs)

force_tick_labels(ax_scatter, 'x')
force_tick_labels(ax_scatter, 'y')
force_tick_labels(ax_histx, 'y')
force_tick_labels(ax_histy, 'x')


fig.canvas.draw()

def move_label(tick, dx_pt, dy_pt):
    offset = ScaledTranslation(dx_pt/72, dy_pt/72, fig.dpi_scale_trans)
    tick.label1.set_transform(tick.label1.get_transform() + offset)

def lengthen(tick, size=SZ_TICKLEN, width=LW_TICKLEN):
    tick.tick1line.set_markersize(size)
    tick.tick1line.set_markeredgewidth(width)

sc_xticks = ax_scatter.xaxis.get_major_ticks()
hy_xticks = ax_histy.xaxis.get_major_ticks()

if sc_xticks:
    lengthen(sc_xticks[-1])
    move_label(sc_xticks[-1], -30, 0)
if hy_xticks:
    lengthen(hy_xticks[0])
    move_label(hy_xticks[0], 30, 0)

hx_yticks = ax_histx.yaxis.get_major_ticks()
sc_yticks = ax_scatter.yaxis.get_major_ticks()

if hx_yticks:
    lengthen(hx_yticks[0])
    move_label(hx_yticks[0], 0, 30)
if sc_yticks:
    lengthen(sc_yticks[-1])
    move_label(sc_yticks[-1], 0, -30)


fig.canvas.draw()
pos_sc = ax_scatter.get_position()
pos_hx = ax_histx.get_position()
pos_hy = ax_histy.get_position()

x_left = pos_sc.x0 - 0.090
fig.text(x_left, (pos_hx.y0 + pos_hx.y1) / 2,
         'WPD', rotation=90,
         ha='center', va='center', fontsize=FS_DENS)
fig.text(x_left, (pos_sc.y0 + pos_sc.y1) / 2,
         labels[1], rotation=90,
         ha='center', va='center', fontsize=FS_LABEL)

y_bottom = pos_sc.y0 - 0.080
fig.text((pos_sc.x0 + pos_sc.x1) / 2, y_bottom,
         labels[0],
         ha='center', va='center', fontsize=FS_LABEL)
fig.text((pos_hy.x0 + pos_hy.x1) / 2, y_bottom,
         'WPD',
         ha='center', va='center', fontsize=FS_DENS)


plt.show()