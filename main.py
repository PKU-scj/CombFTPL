from Environment import Env
from ALGS.FTPL import FTPL
from ALGS.CombUCB import CombUCB
from ALGS.TS import TompsonSampling
from ALGS.EXP2 import EXP2
from ALGS.LOGBAR import LOGBAR
from ALGS.HYBRID import HYBRID
from utils import draws
import numpy as np
import os
import time

def test_method(name, algorithm_class, env_setting, parameters, epoches, start_round=0):
    env = Env(time_horizon = parameters["time_horizon"],
              num_arms = parameters["num_arms"],
              num_action = parameters["num_action"],
              Delta = parameters["Delta"],
              way = env_setting)
    saved_path = f"./results_{env_setting}/T_{parameters['time_horizon']}"
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    for i in range(epoches):
        env.reset(random_seed=2025)
        algorithm = algorithm_class(time_horizon = parameters["time_horizon"],
                                    num_arms = parameters["num_arms"],
                                    num_action = parameters["num_action"])
        actions = algorithm.run(env)
        if env_setting == "stochastic":
            all_losses = np.sum(actions * np.broadcast_to(env.arm_means, (env.time_horizon, env.num_arms)), axis=1)
        else:
            all_losses = np.sum(actions * env.arms, axis=1)
        losses = np.cumsum(all_losses, axis=0)
        bm = env.benchmark()
        reg = losses - bm
        np.save(f"{saved_path}/regret_{name}_r_{start_round+i}.npy", reg)
if __name__ == "__main__":
    PARAMETERS = {
        "time_horizon": 1 * 10 ** 7,
        "num_arms": 10,
        "num_action": 5,
        "Delta": 0.1,
    }
    METHODS = ["FTPL","CUCB","TS","EXP2","LOGBAR","HYBRID"]
    ENV_SETTINGS = ['stochastic', 'adversarial']
    EPOCHES = 20
    START_ROUND = 0
    METHOD_MAP = {
        "FTPL": FTPL,
        "CUCB": CombUCB,
        "TS": TompsonSampling,
        "EXP2": EXP2,
        "LOGBAR": LOGBAR,
        "HYBRID": HYBRID,
    }
    for name in METHODS:
        for env_setting in ENV_SETTINGS:
            test_method(name, METHOD_MAP[name], env_setting, PARAMETERS, epoches=EPOCHES,start_round=START_ROUND)

