import luigi

# Here we are importing our own tasks, provided they are 
# arranged in a python module (folder) named "g2luigi"

from root_to_boost import rootToBoost
from global_params import GlobalParams
from g2_fit import g2FitHist
from si import flatten2dArray

import matplotlib.pyplot as plt

from g2Fitter import *

# ----------------------------------------------------------------------
# DEFINE THE MAIN WORKFLOW DEPENDENCY GRAPH
# We can overload the main functions to pass the energy range as input
# ----------------------------------------------------------------------

#overload the fitter so that it passes the 
class g2FitHistEnergyScan(g2FitHist):
  axisCuts = luigi.Parameter(default=None, description="['axis string', lower bound, upper bound] -> ['y', 1700, 3200]")
  def requires(self):
    return rootToBoost(axisCuts=self.axisCuts) 

class energyScan(law.Task):

    eLow = luigi.FloatParameter(default=1000.)
    eHigh = luigi.FloatParameter(default=3000.)
    eWidth = luigi.FloatParameter(default=100.)

    def requires(self):
        for energy in np.arange(self.eLow, self.eHigh, self.eWidth):
            thiscut = str([['y',energy, energy+self.eWidth]])
            print(thiscut)
            yield g2FitHistEnergyScan(axisCuts=thiscut)

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+GlobalParams().campaignName+"_output_hist_energyScan.pickle")
    
    def run(self):
        input = self.input()
        print("Raw input:", input)

        fig,axs = plt.subplots(int(np.ceil(len(input)/4)), 4,figsize=(15,25), sharex=True, sharey=False)
        ax = flatten2dArray(axs)

        hists = []
        for i, inputi in enumerate(input):
            print("Input for the second task:", inputi.path)

            opened_hist = pickle.load(open(inputi.path,"rb"))

            histName = list(opened_hist)[0]

            print(opened_hist, histName)

            thisHist = opened_hist[ histName ]

            print(thisHist)
            print(thisHist.x)

            plt.sca(ax[i])
            plt.title(str(thisHist.axisCuts))
            print(thisHist.values)
            ding = plt.plot(thisHist.x, thisHist.y, label="Data")
            dong = plt.plot(thisHist.x, thisHist.fit_function(thisHist.x, thisHist.values), label="Fit")
            # dong=3
            hists.append([ding,dong])
            plt.legend(ncol=2)
            plt.yscale("log")
            plt.grid()

        plt.show()

        data = {'plot':(fig,ax), 'hists':hists }
        self.output().dump(data, formatter="pickle")