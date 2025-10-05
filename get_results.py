
import os
import sys

from summarize import summarize

summarize("logs/cartpole-ablation", 
	"results/cartpole-ablation", (1, 2), smooth_ratio=0.05)

summarize("logs/stac-fixed-pessimism", 
	"results/stac-fixed-pessimism", (3, 3))

#summarize("logs/stac-dropout", 
#	"results/stac-dropout", (3, 3))

summarize("logs/main-comparison", 
	"results/main-comparison", (3, 3))
