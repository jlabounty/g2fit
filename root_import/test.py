import os
import ROOT as r
import boost_histogram as bh 
import pickle

ding = pickle.load(open("./data/test_clustersAndCoincidences_corrected.pickle", "rb"))
h = ding['clustersAndCoincidences/corrected']

print(h.view().value)