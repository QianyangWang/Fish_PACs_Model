import numpy as np
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
from SALib.analyze import pawn
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
import joblib


# 0:edoc, 1:nqw, 2:uf, 3:af, 4:logBAF', 5:logBSAF, 6:rsed, 7:s, 8:logcKlw, 9:logKmn
lb = np.array([0.0, 0.4, 0.01, 0.3, 3.5, -2.0, 0.5, 1, -2, -3])
ub = np.array([0.1, 0.8, 0.05, 0.7, 5.5, 0.0, 0.8, 90, 2, 0])

problem1 = {
    'num_vars': 10, # dim
    'names': ["e_doc", "nqw","uf","af","log(BAF)","log(BSAF)","rsed","s","log(cKlw)","log(Km,N)"],
    'bounds': [[0.0, 0.1],      # edoc         0
               [0.4, 0.8],      # nqw         1
               [0.01, 0.05],      # uf         2
               [0.3, 0.7],    # af            3
               [3.5,5.5],       # logBAF'          4
               [-2.0,0.0],       # logBSAF           5
               [0.5,0.8],       # rsed         6
               [1,90],          # s           7
               [-2,2],     # logcKlw          8
               [-3,0]]      # logKmn          9]          # logKm

}


def index_of_agreement_modified(obs, pred):

    obs = np.array(obs)
    pred = np.array(pred)
    mean_obs = np.mean(obs)
    numerator = np.sum(np.abs((obs - pred)))
    denominator = np.sum(np.abs((np.abs(obs - mean_obs) + np.abs(pred - mean_obs))))

    if denominator == 0:
        return 1.0

    return 1-(numerator / denominator)


def evaluate(n,cases,calinum):

    obss = []
    ress = []
    resall = []
    for c in cases:
        res = joblib.load(r"Results\SLSC\{}.res".format(c))
        obs = pd.read_excel(r"Inputs\Fish\SLSC\{}.xlsx".format(c))
        date = pd.to_datetime(obs["Date"].values[0])
        dates = pd.to_datetime(res[0]["time"])
        idx = int(np.where(dates == date)[0])
        fishc = [res[i]["C_fish"][idx] for i in range(len(res))]
        fishcts = [res[i]["C_fish"] for i in range(len(res))]
        avgobs = np.mean(obs["PC4"].values)
        nfish = len(obs)
        avgsres = [np.average(fishc[i * nfish:i * nfish + nfish]) for i in range(n)]
        obss.append(avgobs)
        ress.append(np.array(avgsres).reshape(-1, 1))
        resall.append(fishcts)
    obss = np.array(obss)
    refmt = np.concatenate(ress, axis=1)
    fitness = np.array([index_of_agreement_modified(obss[0:calinum], refmt[i, 0:calinum].flatten()) for i in range(n)])
    ress = np.array(ress)
    conc = np.average(ress)

    return fitness,conc


simcases = ["Firebag_Middle","Firebag_Upper","HighHill1","Muskeg","STB_Middle1","STB_Upper1","Tar1","STB_Lower","HighHill2","STB_Middle2","STB_Upper2","Tar2"]
glue_samples = np.load(r"Results\SLSC\ParamsSLSC_PC4.npy")
fitness, conc = evaluate(10000, simcases[0:8], calinum=8)

Si = pawn.analyze(problem1, glue_samples, fitness)

df = pd.DataFrame({
    'Parameter': Si['names'],
    'Mean':      Si['mean'],
    'Median':    Si['median'],
    'Min':       Si['minimum'],
    'Max':       Si['maximum'],
    'CV':        Si['CV'],
}).sort_values('Mean', ascending=False).reset_index(drop=True)

print("=" * 60)
print("PAWN Sensitivity Analysis — Summary")
print("=" * 60)
print(df.to_string(index=False, float_format='%.4f'))



