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
        for elow in range(500,2300,100):
            yield rootToBoost(axisCuts="[['y', "+str(elow)+", "+str(elow+100)+"]]")

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+"output_plots.pickle")

    def run(self):
        input = self.input()
        print("Raw input:", input)

        fig,ax = plt.subplots(figsize=(15,5))

        for i, inputi in enumerate(input):
            print("Input for the second task:", inputi.path)

            opened_hist = pickle.load(open(inputi.path,"rb"))

            histName = list(opened_hist)[0]

            print(opened_hist, histName)

            thisHist = opened_hist[ histName ]

            # print(thisHist)

            
            plt.plot(thisHist.axes.centers[0], thisHist.view().value,label=str(i))
        plt.legend(ncol=4)
        plt.yscale("log")
        plt.grid()
        plt.show()

        data = {histName:(fig,ax)}
        self.output().dump(data, formatter="pickle")
        


