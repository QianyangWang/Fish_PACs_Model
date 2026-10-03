import numpy as np
import pandas as pd
from scipy import interpolate


class WaterModel:

    def __init__(self, logkoc, logkdoc, f_oc_TSS):
        self.logkoc = logkoc
        self.logkdoc = logkdoc
        self.f_oc_TSS = f_oc_TSS

    def simulate(self, waterTS, tssTS, docTS, TimeIndex):
        res_dissolved = []
        res_doc = []
        res_particle = []
        for d in range(len(TimeIndex)):
            C_dissolved, C_doc_bound, C_particle = self.calculate_phase_concentrations(
                waterTS[d], tssTS[d], docTS[d]
            )
            res_dissolved.append(C_dissolved)
            res_doc.append(C_doc_bound)
            res_particle.append(C_particle)
        return {
            "time": TimeIndex,
            "Dissolved": np.array(res_dissolved),
            "DOC_Bound": np.array(res_doc),
            "Particle_Bound": np.array(res_particle),
        }

    def simulate_fraction(self, waterTS, tssTS, docTS, TimeIndex):
        res_dissolved = []
        res_doc = []
        res_particle = []
        for d in range(len(TimeIndex)):
            C_dissolved, C_doc_bound, C_particle = self.calculate_phase_concentrations(
                waterTS[d], tssTS[d], docTS[d]
            )
            f_doc_b = C_doc_bound / (C_dissolved + C_doc_bound + C_particle)
            f_particle_b = C_particle / (C_dissolved + C_doc_bound + C_particle)
            f_dissolved = 1 - f_doc_b - f_particle_b
            res_dissolved.append(f_dissolved)
            res_doc.append(f_doc_b)
            res_particle.append(f_particle_b)
        res = pd.DataFrame(
            {
                "time": TimeIndex,
                "Dissolved_Fraction": np.array(res_dissolved),
                "DOC_Bound_Fraction": np.array(res_doc),
                "Particle_Bound_Fraction": np.array(res_particle),
            }
        )
        res = res.set_index("time")
        return res

    def calculate_phase_concentrations(self, C_w, TSS_mgL, DOC_mgL):

        Koc = 10 ** self.logkoc # L/kg
        Kdoc = 10 ** self.logkdoc # L/kg

        # Eq SI-S1
        Kp = self.f_oc_TSS * Koc  # L/kg_ss

        TSS_kgL = TSS_mgL * 1e-6  # mg_ss/L -> kg_ss/L
        DOC_kgL = DOC_mgL * 1e-6  # mg_oc/L -> kg_oc/L

        # Eq SI-S2
        DOC_contrib = Kdoc * DOC_kgL
        TSS_contrib = Kp * TSS_kgL
        denominator = 1 + DOC_contrib + TSS_contrib
        C_dis = C_w / denominator  #Please note variable C_w here is C_dis in equation
        C_doc_bound = Kdoc * C_dis * DOC_kgL
        C_particle = Kp * C_dis * TSS_kgL
        C_dissolved = C_w - C_doc_bound - C_particle

        return C_dissolved, C_doc_bound, C_particle


class Fish:

    def __init__(
        self,
        M,
        lipid,
        eff_dissolved,
        eff_doc_b,
        eff_particle_b,
        uf,
        kas,
        kqw=1.4,
        nqw=0.6,
        kql=0.01,
        iniC=0,
    ):
        self.M = M          # g
        self.lipid = lipid  # %
        self.L = lipid / 100 # fraction
        self.Vl = self.M * self.L / 1000  # g/1000 = L assuming 1000 g/L
        self.Vf = self.M / 1000
        self.Qw = self.cal_Qw(self.M, kqw, nqw)  # L/d
        self.Ql = self.cal_Ql_v2(self.Qw, kql) # L/d

        self.uf = uf  # g/gfish_day
        self.kas = kas

        self.eff_dissolved = eff_dissolved
        self.eff_doc_b = eff_doc_b
        self.eff_particle_b = eff_particle_b

        self.iniC = iniC

    def cal_Qw(self, M, kw=1.4, n=0.6):
        """
        :param M: fish mass (g)
        :param kw: ventilation rate coefficient
        :param n: allometric scaling exponent
        :return: Qw (L/d)
        """
        Qw = kw * M ** n
        return Qw

    def cal_Ql(self, Vl, kl=10):
        """
        Discarded very beginning Gobas Model version
        :param Vl: lipid volume (L)
        :param kl: lipid processing rate coefficient (calibratable)
        :return: Ql (L/day)
        """
        Ql = kl * Vl
        return Ql

    def cal_Ql_v2(self, Qw, kl=0.01):
        """
        Discarded very beginning Gobas Model version
        :param Vl: lipid volume (L)
        :param kl: lipid processing rate coefficient (calibratable)
        :return: Ql (L/day)
        """
        Ql = kl * Qw
        return Ql


class FishModel:

    def __init__(self, start_date, end_date):
        self.simdays = pd.date_range(start_date, end_date)
        self.days = len(self.simdays)

    def _get_value_at_time(self, t, time_series):
        """Get concentration value at a given time step index."""
        idx = min(int(t), len(time_series) - 1)
        return time_series[idx]

    def _rolling_avg_cw(self, C_water, t, window):
        """
        Compute a backward-looking rolling average of C_water up to time step t,
        over a window of `window` days. Represents prey organism accumulation lag.

        For time steps near the left boundary (t < window), the missing preceding
        values are treated as NaN and nanmean is used, so the average is computed
        over however many valid values exist.

        :param C_water: array-like, full water concentration time series (ng/L)
        :param t: current time (days, float)
        :param window: int, number of days to look back (calibratable)
        :return: float, rolling mean Cw for dietary waterborne exposure
        """
        idx = int(t)
        start = idx - window + 1

        if start >= 0:
            segment = list(C_water[start: idx + 1])
        else:
            # Pad missing left-boundary values with NaN
            pad = [np.nan] * abs(start)
            segment = pad + list(C_water[0: idx + 1])

        return np.nanmean(segment)

    def calc_k1ke(self, fish, Kow, logcKlw,logKmN,T):
        """
        Calculate gill uptake (k1) and elimination (ke) rate constants. Elimination here denotes passive + metabolism.

        :param fish: Fish instance
        :param Kow: octanol-water partition coefficient (L/kg or L/L)
        :param logKmN: log10 of the normalized metabolism rate
        :param T: water temperature
        :return: k1, ke (1/d)
        """
        # ke = k2 + kM
        # 1/k1 = (Vf/Qw) + (Vf/Ql)/Klipidw -> Vf/Qw + Vf/(Ql * Klipidw) -> Vf * [1/Qw + 1/(Ql * Klipidw)]
        # 1/k2 = (Vl/Qw) * Klipidw + Vl/Ql  ->  Vl * (Klipidw/Qw + 1/Ql)
        # kmi = KmN / (Mn/M)^-0.25 / exp[0.01(TN-T)] -> KmN * (Mn/M)^0.25 / exp[0.01(TN-T)]     in  Eq(1-5)
        Klipidw = (10**logcKlw) * Kow
        # k1 = 1 / (fish.Vl * (1 / fish.Qw + 1 / (fish.Ql * Klipidw)) / fish.L)   # Version in Gobas 1987  Vl/L is equivalent to Vf
        k1 = 1 / (fish.Vf * (1 / fish.Qw + 1 / (fish.Ql * Klipidw)))              # Version in Gobas & MacKay 1988  Vf is used instead
        k2 = 1 / (fish.Vl * (Klipidw / fish.Qw + 1 / fish.Ql))
        kmi = (10**logKmN) / ((10/fish.M)**-0.25) / np.exp((0.01*(15-T)))
        ke = k2 + kmi # Eq (1-5)   ; 10 is the standard fish weight 10g in Arnot 2008, 15 is the standard temperature

        return k1, ke

    def calc_k1ke_simplify(self, fish, Kow, logcKlw,logKmN):
        """
        Calculate gill uptake (k1) and elimination (k2) rate constants.

        :param fish: Fish instance
        :param Kow: octanol-water partition coefficient (L/kg or L/L)
        :param coeff: correction coefficient for Zratio
        :param logKmN: log10 of the normalized metabolism rate
        :param T: water temperature
        :return: k1, k2 (1/d)
        """
        Klipidw = (10**logcKlw) * Kow
        k1 = 1 / (fish.Vl * (1 / fish.Qw + 1 / (fish.Ql * Klipidw)) / fish.L)
        ke = 1 / (fish.Vl * (Klipidw / fish.Qw + 1 / fish.Ql)) + (10**logKmN) / ((10/fish.M)**-0.25)

        return k1, ke

    def cal_cfish(
        self,
        k1, ke,
        uf, af,
        cw,
        fd, fdoc, ftss,
        edis, edoc, etss,
        csed, logBSAF, logBAF, rsed,
        c0,
        f_oc_sed,
        f_lipid_prey = 0.03,
        dt=1,
        cw_diet=None,
    ):
        """
        Calculate fish body concentration for one time step.

        :param k1: gill uptake rate constant (1/d)
        :param k2: elimination rate constant (1/d)
        :param uf: feeding rate (g_food/g_fish/d)
        :param kas: food assimilation efficiency  (-)
        :param cw: instantaneous water concentration (ng/L) — used for gill uptake
        :param fd: dissolved fraction of cw  (-)
        :param fdoc: DOC-bound fraction of cw  (-)
        :param ftss: particle-bound fraction of cw  (-)
        :param ed: gill uptake efficiency for dissolved phase (-)
        :param edoc: gill uptake efficiency for DOC-bound phase (-)
        :param etss: gill uptake efficiency for particle-bound phase (-)
        :param cfood: sediment/solid food concentration (ng/g)
        :param logBSAF: biota sediment accumulation factor
        :param logBAF: bio-accumulation factor
        :param rsed: ratio of sediment food contribution
        :param c0: fish concentration at start of time step (ng/L internal units)
        :param dt: time step (days)
        :param cw_diet: rolling-averaged water concentration (ng/L) for dietary
                        waterborne pathway. If None, falls back to instantaneous cw.
        :return: Cf_ng (ng/g), contributions (dict)
        """
        # Fall back to instantaneous cw if no rolling average provided
        if cw_diet is None:
            cw_diet = cw
        # 1. Gill uptake (instantaneous cw — direct contact with water)
        # Cgu(t) = Cw * (fd * edis + fdoc * edoc + fp * ep) * k1/ke * [1 - exp(-ke * t)]                    Eq(1-1)
        gill_dis   = cw * fd   * edis   * (k1 / ke) * (1 - np.exp(-ke * dt))  # ng/L
        gill_doc   = cw * fdoc * edoc * (k1 / ke) * (1 - np.exp(-ke * dt))    # ng/L
        gill_tss   = cw * ftss * etss * (k1 / ke) * (1 - np.exp(-ke * dt))    # ng/L
        gill_total = gill_dis + gill_doc + gill_tss

        # 2. Dietary uptake
        # food_uptake1: sediment-associated prey (uses cfood directly)
        # food_uptake2: waterborne prey (uses rolling-averaged cw_diet)
        # Csf(t) = [(Csed/soc * BSAF * prey_lipid) * rsed * uf * af] * 1000/ke * (1 - np.exp(-ke * dt))     Eq(1-2)
        # Cwf(t) = [(Cw_bar * BAF') * (1-rsed) * uf * af] * 1/ke * (1 - np.exp(-ke * dt))                   Eq(1-3)
        BSAF = 10**logBSAF
        fBAF = (10**logBAF)/1000  # L/kg -> L/g  unit converted for convenience, making it equivalent to 1/ke in Eq(1-3)

        food_uptake1 = ((csed  * BSAF / f_oc_sed * f_lipid_prey) *   rsed  * uf * af) * 1000 / ke * (1 - np.exp(-ke * dt)) # ng/g * g/g_day -> ng/g_day; ng/g_day * 1000 -> ng/L_day; ng/L_day/(1/day) -> ng/L
        food_uptake2 = ((cw_diet * fBAF) * (1 - rsed) * uf * af) * 1000 / ke * (1 - np.exp(-ke * dt))    # ng/L * L/g -> ng/g; ng/g * g/g_day -> ng/g_day; ng/g_day * 1000 -> ng/L_day; ng/L_day/(1/day) -> ng/L
        food_uptake  = food_uptake1 + food_uptake2

        # 3. Initial residue (decay of previous time step concentration)
        initial_residue = c0 * np.exp(-ke * dt)                                   # ng/L                    Eq(1-4)

        # Total fish concentration (ng/L internal units)
        Cf = gill_total + food_uptake + initial_residue                           # ng/L

        # Contribution percentages (excluding initial residue)
        total_contrib = gill_total + food_uptake
        uptake_pct = ((gill_total + food_uptake) / Cf) * 100 if Cf > 0 else 0

        if total_contrib > 0:
            gill_dis_pct = (gill_dis    / total_contrib) * 100
            gill_doc_pct = (gill_doc    / total_contrib) * 100
            gill_tss_pct = (gill_tss    / total_contrib) * 100
            food_s_pct   = (food_uptake1 / total_contrib) * 100
            food_w_pct   = (food_uptake2 / total_contrib) * 100
        else:
            gill_dis_pct = gill_doc_pct = gill_tss_pct = food_s_pct = food_w_pct = 0

        contributions = {
            "gill_dis_pct": gill_dis_pct,
            "gill_doc_pct": gill_doc_pct,
            "gill_tss_pct": gill_tss_pct,
            "food_s_pct":   food_s_pct,
            "food_w_pct":   food_w_pct,
            "uptake_pct":   uptake_pct,
        }

        Cf_ng = Cf / 1000  # ng/L -> ng/g
        return Cf_ng, contributions

    def solve_concentration(
        self,
        fish,
        kow,
        logcKlw,
        C_water,
        C_sed,
        logBSAF,
        logBAF,
        rsed,
        f_dissolved,
        f_doc_b,
        f_particle_b,
        prey_window,
        logKm,
        wT,
        f_oc_sed,
        dt=None,
    ):
        """
        Simulate fish body concentration over the model period.

        :param fish: Fish instance
        :param kow: octanol-water partition coefficient
        :param C_water: array, total water concentration time series (ng/L)
        :param C_sed: array, sediment concentration time series (ng/g)
        :param logBSAF: food BSAF
        :param logBAF: food BAF
        :param rsed: sediment food ratio parameter
        :param f_dissolved: array, dissolved fraction of C_water time series
        :param f_doc_b: array, DOC-bound fraction time series
        :param f_particle_b: array, particle-bound fraction time series
        :param wT: water temperature time series
        :param dt: time step (days); defaults to 1.0
        :param prey_window: int, rolling average window (days) for dietary
                            waterborne Cw — represents prey accumulation
                            timescale. Calibratable parameter. Default = 7 days.
                            Recommended range for PAHs: 3–21 days.
        :return: dict with time series of C_fish and contribution percentages
        """
        if dt is None:
            dt = 1.0

        prey_window = int(prey_window)

        total_days = self.days
        num_steps = int(total_days / dt) + 1
        dt = total_days / (num_steps - 1)  # adjust to exactly cover simulation period

        #k1, ke = self.calc_k1ke_simplify(fish, kow, coeff,logKm)

        kas = fish.kas
        uf  = fish.uf
        ed   = fish.eff_dissolved
        edoc = fish.eff_doc_b
        etss = fish.eff_particle_b

        all_times  = []
        all_C_fish = []

        Cfish = fish.iniC * 1000  # ng/g -> ng/L (internal units)

        Contrib = {
            "gill_dis":       np.nan,
            "gill_doc":       np.nan,
            "gill_tss":       np.nan,
            "gill_total":     np.nan,
            "food_uptake":    np.nan,
            "initial_residue":np.nan,
            "gill_dis_pct":   np.nan,
            "gill_doc_pct":   np.nan,
            "gill_tss_pct":   np.nan,
            "food_s_pct":     np.nan,
            "food_w_pct":     np.nan,
            "uptake_pct":     np.nan,
        }

        gill_dis_pct_arr = []
        gill_doc_pct_arr = []
        gill_tss_pct_arr = []
        food_s_pct_arr   = []
        food_w_pct_arr   = []
        uptake_pct_arr   = []

        for step in range(num_steps):
            t = step * dt

            all_times.append(t)
            all_C_fish.append(Cfish / 1000) # ng/L -> ng/g assume 1.0 kg/L
            gill_dis_pct_arr.append(Contrib["gill_dis_pct"])
            gill_doc_pct_arr.append(Contrib["gill_doc_pct"])
            gill_tss_pct_arr.append(Contrib["gill_tss_pct"])
            food_s_pct_arr.append(Contrib["food_s_pct"])
            food_w_pct_arr.append(Contrib["food_w_pct"])
            uptake_pct_arr.append(Contrib["uptake_pct"])

            if step < num_steps - 1:
                C_w = self._get_value_at_time(t, C_water)
                C_s = self._get_value_at_time(t, C_sed)
                f_d   = self._get_value_at_time(t, f_dissolved)
                f_doc = self._get_value_at_time(t, f_doc_b)
                f_p   = self._get_value_at_time(t, f_particle_b)
                tmp = self._get_value_at_time(t,wT)
                poc = self._get_value_at_time(t,f_oc_sed)
                # update k2 for different temperature
                k1, ke = self.calc_k1ke(fish, kow, logcKlw, logKm,tmp)

                # Rolling-averaged Cw for dietary waterborne pathway
                C_w_diet = self._rolling_avg_cw(C_water, t, window=prey_window)
                Cfish_new, Contrib = self.cal_cfish(
                    k1, ke,
                    uf, kas,
                    C_w,        # instantaneous cw → gill uptake
                    f_d, f_doc, f_p,
                    ed, edoc, etss,
                    C_s, logBSAF, logBAF, rsed,
                    c0=Cfish,
                    f_oc_sed=poc,
                    dt=dt,
                    cw_diet=C_w_diet,  # rolling avg cw → dietary waterborne
                )
                Cfish = Cfish_new * 1000    # method cal_cfish returns ng/g; ng/g * 1000g/L -> ng/L

        # Convert to numpy arrays
        C_fish_arr       = np.array(all_C_fish)
        gill_dis_pct_arr = np.array(gill_dis_pct_arr)
        gill_doc_pct_arr = np.array(gill_doc_pct_arr)
        gill_tss_pct_arr = np.array(gill_tss_pct_arr)
        food_s_pct_arr   = np.array(food_s_pct_arr)
        food_w_pct_arr   = np.array(food_w_pct_arr)
        uptake_pct_arr   = np.array(uptake_pct_arr)

        # Interpolate to daily output if dt != 1
        C_f      = interpolate.interp1d(all_times, C_fish_arr,       kind="linear", fill_value="extrapolate")
        Dis_f    = interpolate.interp1d(all_times, gill_dis_pct_arr,  kind="linear", fill_value="extrapolate")
        Doc_f    = interpolate.interp1d(all_times, gill_doc_pct_arr,  kind="linear", fill_value="extrapolate")
        Tss_f    = interpolate.interp1d(all_times, gill_tss_pct_arr,  kind="linear", fill_value="extrapolate")
        Food_s_f = interpolate.interp1d(all_times, food_s_pct_arr,    kind="linear", fill_value="extrapolate")
        Food_w_f = interpolate.interp1d(all_times, food_w_pct_arr,    kind="linear", fill_value="extrapolate")
        Uptk_f   = interpolate.interp1d(all_times, uptake_pct_arr,    kind="linear", fill_value="extrapolate")

        target_times = np.arange(self.days)

        C_fish_daily    = C_f(target_times)
        gill_dis_pct    = Dis_f(target_times)
        gill_doc_pct    = Doc_f(target_times)
        gill_tss_pct    = Tss_f(target_times)
        food_s_pct      = Food_s_f(target_times)
        food_w_pct      = Food_w_f(target_times)
        uptake_pct      = Uptk_f(target_times)

        return {
            "time":         np.array(self.simdays),
            "C_fish":       C_fish_daily,
            "gill_dis_pct": gill_dis_pct,
            "gill_doc_pct": gill_doc_pct,
            "gill_tss_pct": gill_tss_pct,
            "food_s_pct":   food_s_pct,
            "food_w_pct":   food_w_pct,
            "uptake_pct":   uptake_pct,
            "fpctall":      food_s_pct + food_w_pct,
        }