# test of the new g2Reader functionality

import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import matplotlib.pyplot as plt

import ast

import uproot

from g2Reader import *

infile = "/home/jlab/g-2/omega_a_KS/data/results_clustersAndCoinc_unrandomized_July7_pileup_corrected.root"
histname = "clustersAndCoincidences/corrected"
muLossName = "clustersAndCoincidences/triples"

ding = g2Histogram(infile, histname, muLossName)
# ding.plothist2d(None, None, True, 1, ding.h_Kloss)
# # ding.plothist2d([0,1])
# plt.colorbar()
# plt.show()

ding.plotprojections()
plt.show()

# hi = ding.project(0,[30,500], [1700,6000], None)
# # hi.show()
# plt.plot(hi.axes[0].centers, hi.view().value)  
# plt.show()

print(ding.h.view(flow=True).value[11,11,11], ding.h.view(flow=True).variance[11,11,11] )
