
import os
import sys

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from summarize import summarize

summary_data = summarize("logs/cartpole-ablation", 
	"results/cartpole-ablation", (1, 2))

summary_data.iloc[:, [0,1,2,3,7,12,5,9,13,6,10,11,4,8]]
summary_data.columns[[0,1,2,3,7,12,5,9,13,6,10,11,4,8]]

#summary_data = pd.read_csv("results/cartpole-ablation/summary.csv")

#summary_data.columns = list(map(lambda x: "$"+x.replace("__", "\\quad").replace("@", "\\").replace("~", "/")+"$" , summary_data.columns))

#deterministic_auc_idx = (summary_data.iloc[:, 0] == "CartPoleSwingUp-v1-Deterministic") & (summary_data.iloc[:, 1] == "auc_scores")

#stochastic_index = (summary_data.iloc[:, 0] == "CartPoleSwingUp-v1-Stochastic") & (summary_data.iloc[:, 1] == "auc_scores")


#sns.barplot(data=summary_data[deterministic_auc_idx])
#plt.xticks(rotation=70)
#plt.tight_layout()

