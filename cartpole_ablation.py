
import os 
from typing import List, Tuple, Union, Dict, Any
import jsonlines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import colormaps
import matplotlib.ticker as ticker
import scipy.stats as st
import seaborn as sns


def visualize_cartpole_memory(env: str, confs: List[str], time_intervals: List[Tuple[int]], path: os.PathLike, result_path: os.PathLike, title: str, confinfos: List[str], colormap: str = "turbo"):
    VMIN = 0
    VMAX_XY = 3
    VMAX_RATES = 0.4
    ncol = len(confs)
    nrow = len(time_intervals)
    figsize = (nrow * 3, ncol * 2)
    fig_xy, ax_xy = plt.subplots(ncol, nrow, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    fig_rates, ax_rates = plt.subplots(ncol, nrow, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)

    for i, conf in enumerate(confs):
        for j, (start, end) in enumerate(time_intervals):
            conf_latex = "$"+conf.replace("__", "\\quad").replace("@", "\\").replace("~", "/")+"$"
            env_path = os.path.join(path, env)
            conf_path = os.path.join(env_path, conf)
            obs_memory_list = []
            for k, trial in enumerate(sorted(os.listdir(conf_path))):
                trial_dir = os.path.join(conf_path, trial)
                with open(os.path.join(trial_dir, "observation.npy"), "rb") as f:
                    obs_memory_flat_k = np.load(f, allow_pickle=True)
                    obs_memory_k =  np.stack(obs_memory_flat_k[start:end], axis=0)
                    obs_memory_list.append(obs_memory_k)  
            obs_memory = np.concatenate(obs_memory_list, axis=0)
            # states
            pole = 0.5  # pole length
            theta = np.atan2(obs_memory[:, 2], obs_memory[:, 3])
            thetadot = obs_memory[:, 4]
            pos = obs_memory[:, 0]
            vel = obs_memory[:, 1]
            x = obs_memory[:, 0] - pole*obs_memory[:, 2]  # x = pos - pole*sin(theta)
            y = pole*(obs_memory[:, 3])  # y = pole*cos(theta)
            # state limits
            theta_max = np.pi 
            theta_min = -np.pi
            thetadot_max = 3
            thetadot_min = -3
            x_max = 2.4
            x_min = -2.4
            y_max = pole
            y_min = -pole
            pos_max = 2.4
            pos_min = -2.4
            vel_max = 3
            vel_min = -3

            # x vs y
            x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            #reward_grid = (y_grid/pole +1)/2
            x_y_positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
            x_y_values = np.vstack([x, y])
            x_y_kernel = st.gaussian_kde(x_y_values)
            f_x_y = np.reshape(x_y_kernel(x_y_positions).T, x_grid.shape)
            ax_xy[i, j].set_xlim(x_min, x_max)
            ax_xy[i, j].set_ylim(y_min, y_max)
            if i == 0:
                ax_xy[i, j].set_title(f"{start//1000}K - {end//1000}K", fontsize=10)            
            if i == len(confs) - 1:
                ax_xy[i, j].set_xlabel("$x$")
            if j == 0:                
                ax_xy[i, j].set_ylabel(f"{confinfos[i]} \n$y$")
            countorf_xy = ax_xy[i, j].contourf(x_grid, y_grid, f_x_y, cmap=colormap, vmin=VMIN, vmax=VMAX_XY, levels=np.linspace(VMIN, VMAX_XY, 200), extend='both')
            if j ==  len(time_intervals) - 1:
                cbar_xy = fig_xy.colorbar(countorf_xy, use_gridspec=False)
                cbar_xy.set_ticks([])
            
            # vel vs thetadot
            vel_grid, thetadot_grid = np.mgrid[vel_min:vel_max:100j, thetadot_min:thetadot_max:100j]
            vel_thetadot_positions = np.vstack([vel_grid.ravel(), thetadot_grid.ravel()])
            vel_thetadot_values = np.vstack([vel, thetadot])
            vel_thetadot_kernel = st.gaussian_kde(vel_thetadot_values)
            f_vel_thetadot = np.reshape(vel_thetadot_kernel(vel_thetadot_positions).T, vel_grid.shape)
            ax_rates[i, j].set_xlim(vel_min, vel_max)
            ax_rates[i, j].set_ylim(thetadot_min, thetadot_max)
            if i == 0:
                ax_rates[i, j].set_title(f"{start//1000}K - {end//1000}K", fontsize=10)
            if i == len(confs) - 1:
                ax_rates[i, j].set_xlabel("$\\dot {x}$")
            if j == 0:
                ax_rates[i, j].set_ylabel(f"{confinfos[i]} \n$\\dot{{\\theta}}$")
            countorf_rates = ax_rates[i, j].contourf(vel_grid, thetadot_grid, f_vel_thetadot, cmap=colormap, vmin=VMIN, vmax=VMAX_RATES, levels=np.linspace(VMIN, VMAX_RATES, 200), extend='both')
            if j ==  len(time_intervals) - 1:
                cbar_rates = fig_rates.colorbar(countorf_rates, use_gridspec=False)
                cbar_rates.set_ticks([])
            
    fig_xy.suptitle(f"{title}, {env}: $x$ vs $\\dot{{x}}$")
    fig_rates.suptitle(f"{title}, {env}: $\\theta$ vs $\\dot{{\\theta}}$")
    fig_xy.savefig(os.path.join(result_path, f"x_vs_y_{env}.png"))
    fig_rates.savefig(os.path.join(result_path, f"xdot_vs_thetadot_{env}.png"))

    return fig_xy, fig_rates

if __name__=="__main__":
    # Example usage
    # https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
    # CMRmap
    # inferno
    
    fig_xy, fig_rates = visualize_cartpole_memory(
        "CartPoleSwingUp-v1-Deterministic", 
        ["STAC__@beta=0.25", "STAC__@beta=0.25__@delta_{@pi}=0.01", "STAC__@beta=0.25__@delta_{Q}=0.01", "STAC__@beta=0.25__@delta_{@pi,Q}=0.01"], 
        [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000)],
        "logs/cartpole-ablation", 
        "results/cartpole-ablation", 
        "STAC $\\beta=0.25$", 
        ["No Dropout", "$\\pi$ Dropout", "$Q$ Dropout", "$\\pi$ and $Q$ Dropout"],
        colormap="turbo")

    fig_xy, fig_rates = visualize_cartpole_memory(
        "CartPoleSwingUp-v1-Stochastic", 
        ["STAC__@beta=0.25", "STAC__@beta=0.25__@delta_{@pi}=0.01", "STAC__@beta=0.25__@delta_{Q}=0.01", "STAC__@beta=0.25__@delta_{@pi,Q}=0.01"], 
        [(0, 10000), (10000, 20000), (20000, 30000), (30000, 40000)],
        "logs/cartpole-ablation", 
        "results/cartpole-ablation", 
        "STAC $\\beta=0.25$", 
        ["No Dropout", "$\\pi$ Dropout", "$Q$ Dropout", "$\\pi$ and $Q$ Dropout"],        
        colormap="turbo")
