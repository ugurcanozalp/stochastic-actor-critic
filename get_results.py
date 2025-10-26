
import os
import sys

from summarize import summarize
import numpy as np


summary_pessimism=summarize("logs/stac-fixed-pessimism", 
	"results/stac-fixed-pessimism", (3, 3))

summary_dropout=summarize("logs/stac-dropout", 
	"results/stac-dropout", (3, 3))

summary_comparison=summarize("logs/main-comparison", 
	"results/main-comparison", (3, 3))

#print(summary_comparison.swaplevel(i=1,j=0).loc["avg_scores"].apply(lambda x: np.round(x, 1)) )
#print(summary_comparison.swaplevel(i=1,j=0).loc["std_scores"].apply(lambda x: np.round(x, 1)) )
#print(summary_comparison.swaplevel(i=1,j=0).loc["last_score_iqm"].apply(lambda x: np.round(x, 1)) )
#print(summary_pessimism.swaplevel(i=1,j=0).loc["last_score_iqm"].apply(lambda x: np.round(x, 1)) )
#print(summary_pessimism.swaplevel(i=1,j=0).loc["avg_scores"].apply(lambda x: np.round(x, 1)) )

#summary_pessimism.swaplevel(i=1,j=0).loc['avg_scores'].transpose()
