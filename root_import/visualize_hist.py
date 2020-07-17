import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast
import matplotlib.pyplot as plt

from root_to_boost import rootToBoost
from global_params import GlobalParams

class visualizeInputHist(law.Task):
    '''
        Takes the output from root_to_boost and visualizes it. Just as a test of luigis ability to schedule things.
    '''

    def requires(self):
        return rootToBoost()

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+"output_plots.pickle")

    def run(self):
        input = self.input()
        print("Input for the second task:", input.path)

        opened_hist = pickle.load(open(input.path,"rb"))

        histName = list(opened_hist)[0]

        print(opened_hist, histName)

        thisHist = opened_hist[ histName ]

        # print(thisHist)

        fig,ax = plt.subplots(figsize=(15,5))
        plt.plot(thisHist.axes.centers[0], thisHist.view().value)
        plt.show()

        data = {histName:(fig,ax)}
        self.output().dump(data, formatter="pickle")
        


