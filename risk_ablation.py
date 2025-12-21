
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os 
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator

import scipy.stats as st
import seaborn as sns

sys.path.append("/home/ugurcan/ai/rl/rl-warehouse")

from rlwarehouse import Agent
from rlwarehouse.algos import STAC


def visualize_masspoint_behavior(agents: List[Agent]):
    x_max = 1.0
    x_min = 0.0
    y_max = 1.0
    y_min = 0.0    
    ncol = len(agents)
    figsize = (ncol * 3, 3.5)
    fig_xy, ax_xy = plt.subplots(1, ncol, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)

    obs_memory_list = []
    for i, agent in enumerate(agents):
        for _ in range(100):  # 100 episodes
            agent.eval(record=True, exploit=False)
            obs_memory_list.append(agent._obs_history)
        obs_memory = np.concatenate(obs_memory_list, axis=0)
        # x vs y
        x_grid, y_grid = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        #reward_grid = (y_grid/pole +1)/2
        x_y_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        x_y_values = np.vstack([obs_memory[:, 0], obs_memory[:, 1]])
        cleanmask=np.logical_not(np.isnan(x_y_values)).any(axis=0)
        x_y_kernel = st.gaussian_kde(x_y_values[:, cleanmask])
        f_x_y = np.reshape(x_y_kernel(x_y_positions).T, x_grid.shape)
        ax_xy[i].set_xlim(x_min, x_max)
        ax_xy[i].set_ylim(y_min, y_max)
        ax_xy[i].set_title(f"$\\beta$={agent._beta}", fontsize=10)            
        ax_xy[i].set_xlabel("$x$")
        if i == 0:
            ax_xy[i].set_ylabel(f"$y$")
        countorf_xy = ax_xy[i].contourf(x_grid, y_grid, f_x_y, cmap="turbo", vmin=0, vmax=3, levels=np.linspace(0, 3, 200), extend='both')
        if i == ncol - 1:
            cbar_xy = fig_xy.colorbar(countorf_xy, use_gridspec=False)
            cbar_xy.locator = MaxNLocator(nbins=5)
            cbar_xy.update_ticks()      
        circ=plt.Circle((0.5, 0.5), 0.3, color='w', fill=False)                
        ax_xy[i].add_patch(circ)

    fig_xy.suptitle(f"Position heatmap for $\\beta$: RiskyPointMass-v0", fontsize=16)
    fig_xy.savefig(os.path.join("results/risk-sensitive", "x_vs_y_riskymasspoint.png"))



agent_beta_0 = STAC(target_entropy=-2, autotune=True, beta=0.0, pi_dropout=0.0, q_dropout=0.0, env_name="RiskyPointMass-v0", render_mode="none", start_steps=0)
agent_beta_0.load_ckpt("logs/risk-sensitive/RiskyPointMass-v0/STAC__@beta=0/seed1")
agent_beta_0_125 = STAC(target_entropy=-2, autotune=True, beta=0.125, pi_dropout=0.0, q_dropout=0.0, env_name="RiskyPointMass-v0", render_mode="none", start_steps=0)
agent_beta_0_125.load_ckpt("logs/risk-sensitive/RiskyPointMass-v0/STAC__@beta=0.125/seed1")
agent_beta_25 = STAC(target_entropy=-2, autotune=True, beta=0.25, pi_dropout=0.0, q_dropout=0.0, env_name="RiskyPointMass-v0", render_mode="none", start_steps=0)
agent_beta_25.load_ckpt("logs/risk-sensitive/RiskyPointMass-v0/STAC__@beta=0.25/seed1")
agent_beta_375 = STAC(target_entropy=-2, autotune=True, beta=0.375, pi_dropout=0.0, q_dropout=0.0, env_name="RiskyPointMass-v0", render_mode="none", start_steps=0)
agent_beta_375.load_ckpt("logs/risk-sensitive/RiskyPointMass-v0/STAC__@beta=0.375/seed1")
agent_beta_5 = STAC(target_entropy=-2, autotune=True, beta=0.5, pi_dropout=0.0, q_dropout=0.0, env_name="RiskyPointMass-v0", render_mode="none", start_steps=0)
agent_beta_5.load_ckpt("logs/risk-sensitive/RiskyPointMass-v0/STAC__@beta=0.5/seed1")

agents = [agent_beta_0, agent_beta_0_125, agent_beta_25, agent_beta_375, agent_beta_5]
visualize_masspoint_behavior(agents)