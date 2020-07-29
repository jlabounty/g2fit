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
from g2Fitter import *

import pickle

infile = "/home/jlab/g-2/omega_a_KS/data/results_clustersAndCoinc_unrandomized_July7_pileup_corrected.root"
histname = "clustersAndCoincidences/corrected"
muLossName = "clustersAndCoincidences/triples"

# ding = g2Histogram(infile, histname, muLossName)
# ding.save("./test_save.pickle")
ding = g2Histogram.read_from_pickle(None, "/home/jlab/g-2/g2fit/g2luigi/tests/test_save.pickle")

# ding.plothist2d(None, None, True, 1, ding.h_Kloss)
# # ding.plothist2d([0,1])
# plt.colorbar()
# plt.show()


# xlims=[30,650]
# calo = 1
# whichFit='5par'
# whichCost='LeastSquares'
# initialGuess=[  1.41236293e+07,  6.44282734e+01,  3.76792699e-01, -6.61887714e+01,
#                 5.30832191e+00, ]
# blindingString='wow what a crazy blinding string this is!!!'
# limits=[None, (0.1,1000), None, None, None ]

xlims=[30,650]
calo = 1
whichFit='13par'
whichCost='LeastSquares'
initialGuess=[  1.41236293e+07,  6.44282734e+01,  3.76792699e-01, -6.61887714e+01,
                5.30832191e+00, 0,0,0,
                2.96476900e+00,  0.37,  1.06624461e+00,  1.33384013e+00,
                4.89043459e-01 ]
blindingString='wow what a crazy blinding string this is!!!'
limits=[None, (0.1,1000), None, None, None, None, None,None, (0.1,2000), None,None,None,None ]

# xlims=[30,650]
# calo = 1
# whichFit='17par'
# whichCost='LeastSquares'
# initialGuess=[  1.41236293e+07,  6.44282734e+01,  3.76792699e-01, -6.61887714e+01,
#                 5.30832191e+00, -5.64781236e-01, -4.78112413e-02,  1.92354573e-02,
#                 2.96476900e+00,  4.06198032e-01,  1.06624461e+00,  1.33384013e+00,
#                 4.89043459e-01,
#                 0.0001, 200, 2.3, 0 ]
# blindingString='wow what a crazy blinding string this is!!!'
# limits=[None, (0.1,1000), None, None, None, (-1,1) ,(-1,1), (-1,1), (0.1,1000),None,None,None,None,
#         (-1,1), (0.1,1000), None, None ]

# xlims=[30,650]
# calo = 1
# whichFit='18par'
# whichCost='LeastSquares'
# initialGuess=[  1.41236293e+07,  6.44282734e+01,  3.76792699e-01, -6.61887714e+01,
#                 5.30832191e+00, 0,0,0,
#                 2.96476900e+00,  4.06198032e-01,  1.06624461e+00,  1.33384013e+00,
#                 4.89043459e-01,
#                 0, 200, 2.3, 0,
#                 0 ]

# blindingString='wow what a crazy blinding string this is!!!'
# limits=[None, (0.1,1000), None, None, None, (-1,1) ,(-1,1), (-1,1), (0.1,1000),None,None,None,None,
#         (-1,1), (0.1,1000), None, None, 
#         (0,None) ]


# xlims=[30,650]
# calo = 1
# whichFit='21par'
# whichCost='LeastSquares'
# # initialGuess=[  1.41236293e+07,  6.44282734e+01,  3.76792699e-01, -6.61887714e+01,
# #                 5.30832191e+00, -5.64781236e-01, -4.78112413e-02,  1.92354573e-02,
# #                 2.96476900e+00,  4.06198032e-01,  1.06624461e+00,  1.33384013e+00,
# #                 4.89043459e-01,
# #                 0.0001, 200, 2.3, 0,
# #                 0.0000001,
# #                 0.0001, 100, 3.01  ]
# initialGuess = [56455072.50617181, 64.39436307028377, 0.1078645454458145, -64.02136100347626, 
#                 5.302705714832327, -0.15032346956206255, 0.9965684977225879, 0.3742766554730066, 
#                 8.086297101750757, 0.06309555863826055, 21.48482950372751, 24.60236611838464, 
#                 15.383503962011405, 0.0004243674732773428, 284.8983331626276, 2.2404697325694203, 
#                 2.788860909943641, 14.121353036015451, 0.003472122000239608, 58.907016132111394, 
#                 22.517217625107588]

# blindingString='wow what a crazy blinding string this is!!!'
# limits=[None, (0.1,1000), None, None, None, (-1,1) ,(-1,1), (-1,1), (0.1,1000),None,None,None,None,
#         (-1,1), (0.1,1000), None, None, 
#         None,
#         (-1,1), (0.1,1000), None ]

xlims=[30,650]
calo = 1
whichFit='25par'
whichCost='LeastSquares'
# initialGuess = [56455072.50617181, 64.39436307028377, 0.1078645454458145, -64.02136100347626, 
#                 5.302705714832327, -0.15032346956206255, 0.9965684977225879, 0.3742766554730066, 
#                 8.086297101750757, 0.06309555863826055, 21.48482950372751, 24.60236611838464, 
#                 15.383503962011405, 0.0004243674732773428, 284.8983331626276, 2.2404697325694203, 
#                 2.788860909943641, 14.121353036015451, 0.003472122000239608, 58.907016132111394, 
#                 22.517217625107588, 
#                 0.00001, 200, 1.6, 0.0]
                # ['N', '#tau', 'A', 'R', 
                # '#phi', 'A_{CBO - N}', 'A_{CBO - A}', 'A_{CBO - #phi}', 
                # '#tau_{CBO}', '#omega_{CBO}', '#phi_{CBO - N}', '#phi_{CBO - A}', 
                # '#phi_{CBO - #phi}', 'A_{VW}', '#tau_{VW}', '#omega_{VW}', 
                # '#phi_{VW}', 'K_{loss}', 'A_{2-CBO}', '#tau_{2-CBO}', 
                # '#phi_{2-CBO}', 
                # 'A_{y}', '#tau_{y}', '#omega_{y}', '#phi_{y}']  
                
initialGuess = [56455072.50617181, 64.39436307028377, 0.1078645454458145, -64.02136100347626, 
                5.302705714832327, 0., 0., 0., 
                8.086297101750757, 0.06309555863826055, 21.48482950372751, 24.60236611838464, 
                15.383503962011405, 0., 284.8983331626276, 2.2404697325694203, 
                2.788860909943641, 14.121353036015451, 0., 58.907016132111394, 
                22.517217625107588, 
                0., 200, 1.6, 0.0]

blindingString='wow what a crazy blinding string this is!!!'
# limits=[None, (0.1,1000), None, None, None, (-1,1) ,(-1,1), (-1,1), (0.1,1000),None,None,None,None,
#         (-1,1), (0.1,1000), None, None, 
#         None,
#         (-1,1), (0.1,1000), None,
#         (-1,1), (0.1,1000), None, None ]
limits=[None, (0.1,1000), None, None, None, None, None, None, (0.1,1000),None,None,None,None,
        None, (0.1,1000), None, None, 
        None,
        None, (0.1,1000), None,
        None, (0.1,1000), None, None ]

fitter = g2Fitter(whichFit, whichCost, blindingString, ding.project(0, None, [1700,3200], [1,1]), 
                  initialGuess, xlims, do_iterative_fit=True, fit_list=[5,8,13], fit_limits=limits,
                  useError=False,
                  final_unlimited_fit=False, 
                  triples_hist=ding.h_Kloss[:,bh.loc(calo)])

fitter.do_fit()

# fitter.plot_result()
fig,ax = fitter.draw(labelFit=True)
ax[0].set_yscale("log")
ax[0].set_ylim(10**2, 10**8)
ax[1].set_ylim(-3,3)

# plt.show()

fitter.fft()