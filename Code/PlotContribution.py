import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def calculate_monthly_means(data_dict):
    times = pd.to_datetime(data_dict['time'])
    months = times.month

    variables = ['C_fish', 'gill_dis_pct', 'gill_doc_pct',
                 'gill_tss_pct', 'food_s_pct', 'food_w_pct']

    multiyear_means = {}

    for month in range(1, 13):
        month_mask = months == month
        month_key = f"{month:02d}"
        multiyear_means[month_key] = {}

        for var in variables:
            month_data = data_dict[var][month_mask]
            if len(month_data) > 0:
                multiyear_means[month_key][var] = np.nanmean(month_data)
            else:
                multiyear_means[month_key][var] = np.nan

    return multiyear_means


def weighted_quantile(vals, weights, q):
    vals = np.asarray(vals)
    weights = np.asarray(weights)

    sort_idx = np.argsort(vals)
    sorted_vals = vals[sort_idx]
    sorted_w = weights[sort_idx]
    cum_w = np.cumsum(sorted_w)
    cum_w = cum_w / cum_w[-1]

    i_q = np.searchsorted(cum_w, q, side='left')
    i_q = min(i_q, len(sorted_vals) - 1)
    return sorted_vals[i_q]


def weighted_median(vals, weights):
    return weighted_quantile(vals, weights, 0.5)

###############################

plotcases = ["STB_Lower", "STB_Middle1", "STB_Upper1"]

out_dir = r"Results\SLSC\Behavioral"
table_path = os.path.join(out_dir, "MonthlyPPU.xlsx")
behaviour_samples = np.load(os.path.join(out_dir, "Results\SLSC\Behavioral\BehaviouralParams.npy"))
normalized_weight = np.load(os.path.join(out_dir, "Results\SLSC\Behavioral\BehaviouralWeights.npy"))

################################

assert len(normalized_weight) == behaviour_samples.shape[0], \
    "weight/sample count mismatch"

n_behaviour = behaviour_samples.shape[0]

variables = ['gill_dis_pct', 'gill_doc_pct', 'gill_tss_pct',
             'food_s_pct', 'food_w_pct']
pathway_vars = variables

months = [f"{i:02d}" for i in range(1, 13)]
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

case_data = {}
table_records = []

for c in plotcases:
    contrib = joblib.load(os.path.join(out_dir, f"{c}.res"))
    contrib = [calculate_monthly_means(d) for d in contrib]

    assert len(contrib) % n_behaviour == 0, \
        f"{c}: contrib length {len(contrib)} not divisible by n_behaviour {n_behaviour}"

    nfish = len(contrib) // n_behaviour
    rfmt = [contrib[i * nfish:(i + 1) * nfish] for i in range(n_behaviour)]

    all_contrib = rfmt
    all_weights = normalized_weight

    months_list = [f"{i:02d}" for i in range(1, 13)]

    param_avg = []
    for param_group in all_contrib:
        group_means = {month: {var: [] for var in variables} for month in months_list}

        for fish_data in param_group:
            for month in months_list:
                if month in fish_data:
                    for var in variables:
                        if var in fish_data[month]:
                            group_means[month][var].append(fish_data[month][var])

        month_avg = {}
        for month in months_list:
            month_avg[month] = {}
            for var in variables:
                vals = group_means[month][var]
                month_avg[month][var] = np.mean(vals) if vals else 0
        param_avg.append(month_avg)

    case_data[c] = {}

    for month in months_list:
        case_data[c][month] = {}

        for var in variables:
            weighted_sum = sum(
                all_weights[idx] * pa[month][var]
                for idx, pa in enumerate(param_avg)
            )
            case_data[c][month][var] = weighted_sum

        total = sum(case_data[c][month][var] for var in pathway_vars)

        if total > 0:
            for var in pathway_vars:
                case_data[c][month][var] = case_data[c][month][var] * 100 / total

    for month in months_list:
        totals = np.array([
            sum(pa[month][v] for v in pathway_vars)
            for pa in param_avg
        ])

        for var in pathway_vars:
            vals = np.array([
                pa[month][var] * 100.0 / totals[i] if totals[i] > 0 else 0.0
                for i, pa in enumerate(param_avg)
            ])

            med = weighted_median(vals, all_weights)
            lo95 = weighted_quantile(vals, all_weights, 0.025)
            hi95 = weighted_quantile(vals, all_weights, 0.975)
            lo75 = weighted_quantile(vals, all_weights, 0.125)
            hi75 = weighted_quantile(vals, all_weights, 0.875)
            lo50 = weighted_quantile(vals, all_weights, 0.25)
            hi50 = weighted_quantile(vals, all_weights, 0.75)

            table_records.append({
                "Case": c,
                "Month": month,
                "Variable": var,
                "WeightedMedian": med,
                "PPU50_lo": lo50,
                "PPU50_hi": hi50,
                "PPU75_lo": lo75,
                "PPU75_hi": hi75,
                "PPU95_lo": lo95,
                "PPU95_hi": hi95,
            })

table_df = pd.DataFrame(table_records)
with pd.ExcelWriter(table_path) as writer:
    for c in plotcases:
        sub = table_df[table_df["Case"] == c].drop(columns=["Case"])
        pivot = sub.pivot(index="Month", columns="Variable",
                          values=["WeightedMedian",
                                  "PPU50_lo", "PPU50_hi",
                                  "PPU75_lo", "PPU75_hi",
                                  "PPU95_lo", "PPU95_hi"])
        pivot = pivot.reindex(months)
        pivot.to_excel(writer, sheet_name=c)


colors_pathways = {
    'gill_dis_pct': '#576fa0',
    'gill_doc_pct': '#a7b9d7',
    'gill_tss_pct': '#e6f0fa',
    'food_w_pct': '#fadcb4',
    'food_s_pct': '#e3b78f'
}

plot_months = range(1, 13)
plot_month_labels = ['J', 'F', 'M', 'A', 'M', 'J',
                     'J', 'A', 'S', 'O', 'N', 'D']

for case in plotcases:
    fig, ax = plt.subplots(figsize=(2.8, 2.6))

    dis = [case_data[case][f"{m:02d}"]['gill_dis_pct'] for m in plot_months]
    doc = [case_data[case][f"{m:02d}"]['gill_doc_pct'] for m in plot_months]
    tss = [case_data[case][f"{m:02d}"]['gill_tss_pct'] for m in plot_months]
    foodw = [case_data[case][f"{m:02d}"]['food_w_pct'] for m in plot_months]
    foods = [case_data[case][f"{m:02d}"]['food_s_pct'] for m in plot_months]

    ax.stackplot(plot_months, dis, doc, tss, foodw, foods,
                 labels=['Gill DIS', 'Gill DOC', 'Gill TSS', 'Food (Wat)', 'Food (Sed)'],
                 colors=[colors_pathways['gill_dis_pct'],
                         colors_pathways['gill_doc_pct'],
                         colors_pathways['gill_tss_pct'],
                         colors_pathways['food_w_pct'],
                         colors_pathways['food_s_pct']],
                 alpha=0.8)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.2)

    ax.set_xticks(list(plot_months) + [13.5])
    ax.set_xticklabels(plot_month_labels + ['An.'])

    ax.tick_params(axis='x', which='both', color='black', labelsize=12)
    ax.tick_params(axis='y', which='both', color='black', labelcolor='black', labelsize=12)


    xticklines = ax.get_xticklines()
    if len(xticklines) > 24:
        xticklines[24].set_color('black')

    xtick_labels = ax.get_xticklabels()
    for i, label in enumerate(xtick_labels):
        label.set_color('black')

    cum_6 = 0

    dis_avg = np.mean(dis)
    if dis_avg >= 5:
        ax.text(6.5, cum_6 + dis_avg / 2,
                f'{dis_avg:.1f}%',
                ha='center', va='center', color=colors_pathways['gill_dis_pct'],
                fontweight='bold', fontsize=16,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    cum_6 += dis_avg

    doc_avg = np.mean(doc)
    if doc_avg >= 5:
        ax.text(6.5, cum_6 + doc_avg / 2,
                f'{doc_avg:.1f}%',
                ha='center', va='center', color=colors_pathways['gill_doc_pct'],
                fontweight='bold', fontsize=16,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    cum_6 += doc_avg

    tss_avg = np.mean(tss)
    if tss_avg >= 5:
        ax.text(6.5, cum_6 + tss_avg / 2,
                f'{tss_avg:.1f}%',
                ha='center', va='center', color=colors_pathways['gill_tss_pct'],
                fontweight='bold', fontsize=16,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    cum_6 += tss_avg

    food_w_avg = np.mean(foodw)
    if food_w_avg >= 5:
        ax.text(6.5, cum_6 + food_w_avg / 2,
                f'{food_w_avg:.1f}%',
                ha='center', va='center', color=colors_pathways['food_w_pct'],
                fontweight='bold', fontsize=16,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    cum_6 += food_w_avg

    food_s_avg = np.mean(foods)
    if food_s_avg >= 5:
        ax.text(6.5, cum_6 + food_s_avg / 2,
                f'{food_s_avg:.1f}%',
                ha='center', va='center', color=colors_pathways['food_s_pct'],
                fontweight='bold', fontsize=16, clip_on=False,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])
    cum_6 += food_s_avg

    ax.set_xlim(1, 12)
    ax.set_ylabel('Contribution (%)', fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.label.set_color('black')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{case}.jpg"), dpi=600)
    plt.show()