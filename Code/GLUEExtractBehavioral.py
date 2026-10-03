import numpy as np
import pandas as pd
import joblib
import os


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def index_of_agreement_modified(obs, pred):
    obs = np.array(obs)
    pred = np.array(pred)
    mean_obs = np.mean(obs)

    numerator = np.sum(np.abs((obs - pred)))
    denominator = np.sum(np.abs((np.abs(obs - mean_obs) + np.abs(pred - mean_obs))))

    if denominator == 0:
        return 1.0

    return 1 - (numerator / denominator)


def objfun(n, cases, calinum):
    obss = []
    ress = []
    resall = []
    for c in cases:
        res = joblib.load(r"Results\{}.res".format(c))
        obs = pd.read_excel(r"Inputs\Fish\SLSC\{}.xlsx".format(c))
        date = pd.to_datetime(obs["Date"].values[0])
        dates = pd.to_datetime(res[0]["time"])
        idx = int(np.where(dates == date)[0])
        fishc = [res[i]["C_fish"][idx] for i in range(len(res))]
        fishcts = [res[i]["C_fish"] for i in range(len(res))]
        avgobs = np.mean(obs["PC4"].values)    # P-C4
        nfish = len(obs)
        avgsres = [np.average(fishc[i * nfish:i * nfish + nfish]) for i in range(n)]
        obss.append(avgobs)
        ress.append(np.array(avgsres).reshape(-1, 1))
        resall.append(fishcts)
    obss = np.array(obss)
    refmt = np.concatenate(ress, axis=1)
    fitness = np.array([
        index_of_agreement_modified(obss[0:calinum], refmt[i, 0:calinum].flatten())
        for i in range(n)
    ])
    return 1 - fitness, resall, refmt, obss


def likelihood_uncertainty(samples, fitness, results, top_fraction=0.1, prior_weight=None):
    if prior_weight is None:
        prior_weight = np.ones(len(samples))

    n_total = len(samples)
    n_keep = int(np.ceil(n_total * top_fraction))

    sorted_fitness, index_l = SortFitness(fitness)
    sorted_samples = SortPosition(samples, index_l)
    sorted_results = results[index_l]
    sorted_prior_weight = prior_weight[index_l]

    best_sample = sorted_samples[0]
    best_fitness = sorted_fitness[0]
    best_result = sorted_results[0]

    behaviour_indices = np.array(index_l[:n_keep], dtype=int)
    behaviour_samples = sorted_samples[:n_keep]
    behaviour_fitness = sorted_fitness[:n_keep]
    behaviour_results = sorted_results[:n_keep]
    behavioral_prior = sorted_prior_weight[:n_keep]

    if len(behaviour_samples) == 0:
        print("No behaviour parameter has been found, please adjust the threshold value")
        return

    reciprocals = (1 - behaviour_fitness) * behavioral_prior
    total_likelihood = np.sum(reciprocals)
    normalized_weight = reciprocals / total_likelihood

    sorted_sample_val, sorted_sample_id = np.sort(behaviour_samples, axis=0), np.argsort(behaviour_samples, axis=0)
    id_sample_column = np.array([np.arange(behaviour_samples.shape[1]) for i in range(behaviour_samples.shape[0])])
    normalized_sample_weight = np.array([normalized_weight.flatten() for i in range(behaviour_samples.shape[1])]).T
    normalized_weight_sort = normalized_sample_weight[sorted_sample_id, id_sample_column]
    cum_sample = np.cumsum(normalized_weight_sort, axis=0)

    return (best_sample, best_fitness, best_result,
            behaviour_samples, behaviour_fitness, behaviour_results,
            normalized_weight, normalized_weight_sort, cum_sample,
            index_l, sorted_sample_val, behaviour_indices)


if __name__ == "__main__":

    calicases = ["Firebag_Middle", "Firebag_Upper", "HighHill1", "Muskeg",
                 "STB_Lower", "STB_Middle1", "STB_Upper1", "Tar1"]
    valicases = ["HighHill2", "STB_Middle2", "STB_Upper2", "Tar2"]
    all_cases = calicases + valicases

    res_dir = r"Results\SLSC"
    out_dir = r"Results\SLSC\Behavioral"
    fish_dir = r"Inputs\Fish\SLSC"
    os.makedirs(out_dir, exist_ok=True)

    samples = np.load(os.path.join(res_dir, "ParamsSLSC_PC4.npy"))

    fitness, resall, res, obss = objfun(10000, calicases, calinum=8)

    (best_sample, best_fitness, best_result,
     behaviour_samples, behaviour_fitness, behaviour_results,
     normalized_weight, normalized_weight_sort, cum,
     index_l, sorted_sample_val,
     behaviour_indices) = likelihood_uncertainty(samples, fitness, res, top_fraction=0.1)

    np.save(os.path.join(out_dir, "BehaviourIndices.npy"), behaviour_indices)
    np.save(os.path.join(out_dir, "BehaviouralParams.npy"), samples[behaviour_indices])
    np.save(os.path.join(out_dir, "BehaviouralWeights.npy"), normalized_weight)
    np.save(os.path.join(out_dir, "BehaviouralFitness.npy"), behaviour_fitness)

    for c in all_cases:
        raw_path = os.path.join(res_dir, f"{c}.res")
        if not os.path.exists(raw_path):
            continue

        raw = joblib.load(raw_path)
        fishdf = pd.read_excel(os.path.join(fish_dir, f"{c}.xlsx"))
        n_fish = len(fishdf)
        n_samples = len(raw) // n_fish

        extracted = []
        for s_idx in behaviour_indices:
            if s_idx >= n_samples:
                continue
            base = s_idx * n_fish
            extracted.extend(raw[base: base + n_fish])
        joblib.dump(extracted, os.path.join(out_dir, f"{c}.res"))