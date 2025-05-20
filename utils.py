from math import floor
import matplotlib.pyplot as plt
import numpy as np
from ALGS.FTPL import FTPL
from ALGS.CombUCB import CombUCB
from ALGS.TS import TompsonSampling
from ALGS.EXP2 import EXP2
from ALGS.LOGBAR import LOGBAR
from ALGS.HYBRID import HYBRID
import os
def draws(method_list, env_setting, start_epoch, end_epoch, Time1=10**5, Time2=10**7):
    saved_path  = f"./results_{env_setting}/T_{10**7}"

    colors = ['blue','orange','green','red','purple','brown']
    markers = ['o','s','^','D','v','x']

    step1 = 10
    step2 = 1000
    min_x = 1000
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    mark_place1 = [floor(Time1/10 * i / step1) for i in range(0,10)]
    mark_place2 = [floor(min_x**(1-i/10) * Time2 **(i/10)/step2) for i in range(0,10)]

    for idx, name in enumerate(method_list):
        loading_path = f"{saved_path}/regret_{name}"
        regret = []
        for i in range(start_epoch[name], end_epoch[name]):
            file_path = loading_path + f"_r_{i}.npy"
            if os.path.exists(file_path):
                regret.append(np.load(file_path))
        regret = np.array(regret)
        mean = np.mean(regret, axis=0)
        std = np.std(regret, axis=0)
        mean1 = mean[:Time1:step1]
        std1 = std[:Time1:step1]
        mean2 = mean[:Time2:step2]
        std2 = std[:Time2:step2]
        axs[0].plot(range(0, Time1, step1), mean1, label=name, color=colors[idx], marker=markers[idx], markevery=mark_place1)
        axs[1].plot(range(0, Time2, step2), mean2, label=name, color=colors[idx], marker=markers[idx], markevery=mark_place2)
        if name!="TS" or env_setting != "adversarial":
            axs[0].fill_between(range(0, Time1, step1), mean1-std1, mean1+std1, color=colors[idx], alpha=0.2)
            axs[1].fill_between(range(0, Time2, step2), mean2-std2, mean2+std2, color=colors[idx], alpha=0.2)
    
    axs[0].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[1].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[1].set_xlabel("Time")

    axs[1].set_xscale("log")
    axs[1].set_yscale("log")

    axs[1].set_xlim(min_x, Time2)
    axs[1].set_ylim(1e1)

    axs[0].set_xlim(0, Time1)
    axs[0].set_ylim(0)

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Regret")

    axs[0].legend(loc="upper left")

    save_path = f"./T1_{Time1}_T2_{Time2}_plots"
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(f"{save_path}/{env_setting}_plot.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    ENV_SETTINGS = ['stochastic', 'adversarial']
    METHOD_MAP = {
        "FTPL": FTPL,
        "CUCB": CombUCB,
        "TS": TompsonSampling,
        "EXP2": EXP2,
        "LOGBAR": LOGBAR,
        "HYBRID": HYBRID,
    }
    ST = {"FTPL": 0, "CUCB": 0, "TS": 0, "EXP2": 0, "LOGBAR": 0, "HYBRID": 0}
    ED = {"FTPL": 20, "CUCB": 20, "TS": 100, "EXP2": 20, "LOGBAR": 20, "HYBRID": 20}

    for env_setting in ENV_SETTINGS:
        draws(METHOD_MAP.keys(), env_setting, start_epoch=ST, end_epoch=ED)
        