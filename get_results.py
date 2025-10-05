
import os
import sys

from summarize import summarize

summarize("logs/stac-fixed-pessimism", 
	"results/stac-fixed-pessimism", (3, 3))

#summarize("logs/stac-dropout", 
#	"results/stac-dropout", (3, 3))

summarize("logs/main-comparison", 
	"results/main-comparison", (3, 3))
