
import os
import sys

from summarize import summarize


summarize("all-logs/ablation-stac-dropout-0", 
	"all-results/ablation-stac-dropout-0", (3, 3), figsize=(3*5, 3*3))

summarize("all-logs/ablation-stac-dropout-0.01", 
	"all-results/ablation-stac-dropout-0.01", (3, 3), figsize=(3*5, 3*3))

summarize("all-logs/ablation-stac-dropout-0.02", 
	"all-results/ablation-stac-dropout-0.02", (3, 3), figsize=(3*5, 3*3))

summarize("all-logs/ablation-double-stac", 
	"all-results/ablation-double-stac", (3, 3), figsize=(3*5, 3*3))

summarize("all-logs/dropout-double-comparison", 
	"all-results/dropout-double-comparison", (3, 3), figsize=(3*5, 3*3))

summarize("all-logs/main-comparison", 
	"all-results/main-comparison", (3, 3), figsize=(3*5, 3*3))
