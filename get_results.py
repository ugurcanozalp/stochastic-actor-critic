
import os
import sys

from summarize import summarize


summarize("all-logs/stac-pessimism", 
	"all-results/stac-pessimism", (3, 3))

summarize("all-logs/stac-dropout", 
	"all-results/stac-dropout", (3, 3))

summarize("all-logs/main-comparison", 
	"all-results/main-comparison", (3, 3))
