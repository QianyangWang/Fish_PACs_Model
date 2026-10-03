import time
import numpy as np
import pandas as pd
from pyDOE import lhs
from FishTKModel import Fish,FishModel,WaterModel
import multiprocessing as mp
import joblib


def LHS_sampling(pop,dim,ub,lb,seed=42):
    np.random.seed(seed)
    samples = lb + (ub-lb) * lhs(dim,pop)
    return samples,lb,ub


def create_sediment_daily_series_v1(water_daily_index, sediment_annual_df):
    sediment_daily = pd.DataFrame(index=water_daily_index)
    for col in sediment_annual_df.columns:
        year_to_value = {}
        for date, row in sediment_annual_df.iterrows():
            year_to_value[date.year] = row[col]
        sediment_daily[col] = water_daily_index.year.map(year_to_value)

    return sediment_daily


def create_sediment_daily_series_v2(water_daily_index, sediment_annual_df):

    sediment_daily = pd.DataFrame(index=water_daily_index)

    for col in sediment_annual_df.columns:
        s = sediment_annual_df[col].sort_index()
        s = s[~s.index.duplicated(keep='first')]

        dates = s.index
        values = s.values
        n = len(dates)

        series = pd.Series(index=water_daily_index, dtype=float)

        pre_mask = water_daily_index < dates[0]
        if pre_mask.any():
            series.loc[water_daily_index[pre_mask]] = values[0]

        for i in range(n - 1):
            d_i, v_i = dates[i], values[i]
            d_next, v_next = dates[i + 1], values[i + 1]

            seg_mask = (water_daily_index >= d_i) & (water_daily_index <= d_next)
            seg_dates = water_daily_index[seg_mask]
            if len(seg_dates) == 0:
                continue

            total_seconds = (d_next - d_i).total_seconds()
            if total_seconds > 0:
                frac = (seg_dates - d_i).total_seconds() / total_seconds
                series.loc[seg_dates] = v_i + frac * (v_next - v_i)
            else:
                series.loc[seg_dates] = v_i

        post_mask = water_daily_index > dates[-1]
        if post_mask.any():
            series.loc[water_daily_index[post_mask]] = values[-1]

        sediment_daily[col] = series

    return sediment_daily


def main():

    cali_cases = ["Firebag_Middle","Firebag_Upper","HighHill1","Muskeg","STB_Lower","STB_Middle1","STB_Upper1","Tar1"]
    vali_cases = ["HighHill2","STB_Middle2","STB_Upper2","Tar2"]
    all = cali_cases + vali_cases
    # sampling
    # 0:edoc, 1:nqw, 2:uf, 3:af, 4:logBAF', 5:logBSAF, 6:rsed, 7:s, 8:logcKlw, 9:logKmn
    lb = np.array([0.0,0.4,0.01,0.3,3.5,-2.0,0.5, 1,-2, -3])
    ub = np.array([0.1,0.8,0.05,0.7,5.5, 0.0,0.8,90, 2,  0])

    # P-C4
    Kow = 10**6.46
    logkoc = 6.0
    logkdoc = 5.6

    samples,_,_ = LHS_sampling(10000,10,ub,lb)
    np.save(r"Results\SLSC\ParamsSLSC_PC4.npy",samples)

    for c in all:
        fishdf = pd.read_excel(r"Inputs\Fish\SLSC\{}.xlsx".format(c))
        waterdf = pd.read_excel(r"Inputs\Water\{}.xlsx".format(c),index_col=0)
        try:
            seddf = pd.read_excel(r"Inputs\Sediment\{}.xlsx".format(c),index_col=0)
        except:
            # Assume sediment-associated prey contributing negligible PAC for stations without sediment data
            # HHR and TAR are upstream stations outside the deposit; CAL is used for LKCH, and LKCH is pelagic
            seddf = waterdf.copy()
            seddf[['NC1', 'PC4', 'DC4']] = 0
            seddf['POC'] = 1.0   # avoid 0 division
        waterdf.index = pd.to_datetime(waterdf.index)
        waterts_daily = waterdf["PC4"].resample('D').interpolate(method='linear')
        tssts_daily = waterdf["TSS"].resample('D').interpolate(method='linear')
        docts_daily = waterdf["DOC"].resample('D').interpolate(method='linear')
        tmpts_daily = waterdf["TMP"].resample('D').interpolate(method='linear')
        sed_all = create_sediment_daily_series_v2(waterts_daily.index, seddf)
        sediment_daily = sed_all["PC4"]
        sediment_oc_daily = sed_all["POC"] / 100  # % -> fraction
        wm = WaterModel(logkoc=logkoc, logkdoc=logkdoc, f_oc_TSS=0.03)
        fractions = wm.simulate_fraction(waterts_daily, tssts_daily, docts_daily, waterts_daily.index)

        pool = mp.Pool(processes=5)  # generate n processes
        bthres = []
        for s in samples:
            for f in range(len(fishdf)):
                fsh = fishdf.iloc[f]
                lipid = fsh["Lipid"]
                weight = fsh["Weight"]
                fshobj = Fish(M=weight,
                              lipid=lipid,
                              eff_dissolved=1.0,
                              eff_doc_b=s[0],                                                                           # edoc
                              eff_particle_b=0.0,
                              uf=s[2],                                                                                  # uf
                              kas=s[3],                                                                                 # af
                              kqw=1.4,
                              nqw=s[1],                                                                                 # nqw
                              kql=0.01,
                              iniC=0)
                mdl = FishModel(waterts_daily.index[0], waterts_daily.index[-1])
                bthres.append(pool.apply_async(mdl.solve_concentration, args=(fshobj,                                   # fish
                                                                              Kow,                                      # kow
                                                                              s[8],                                     # logcKlw
                                                                              np.array(waterts_daily),                  # C_water
                                                                              np.array(sediment_daily),                 # C_sed
                                                                              s[5],                                     # logBSAF
                                                                              s[4],                                     # logBAF
                                                                              s[6],                                     # rsed
                                                                              fractions["Dissolved_Fraction"],          # f_dissolved
                                                                              fractions["DOC_Bound_Fraction"],          # f_doc_b
                                                                              fractions["Particle_Bound_Fraction"],     # f_particle_b
                                                                              s[7],                                     # prey_window
                                                                              s[9],                                     # logKm
                                                                              np.array(tmpts_daily),                    # wT
                                                                              np.array(sediment_oc_daily),              # f_oc_sed
                                                                              1.0)))                                    # dt

        pool.close()
        pool.join()
        caseres = []
        for i in range(len(bthres)):
            res = bthres[i].get()
            caseres.append(res)
        joblib.dump(caseres,r"Results\SLSC\{}.res".format(c))


if __name__ == "__main__":
    main()