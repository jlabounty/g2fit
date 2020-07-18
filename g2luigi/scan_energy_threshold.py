import luigi

# Here we are importing our own tasks, provided they are 
# arranged in a python module (folder) named "g2luigi"

from root_to_boost import rootToBoost
from global_params import GlobalParams
from g2_fit import g2FitHist
from si import flatten2dArray

from uncertainties import ufloat
from uncertainties.umath import *

import matplotlib.pyplot as plt

from g2Fitter import *

# ----------------------------------------------------------------------
# DEFINE THE MAIN WORKFLOW DEPENDENCY GRAPH
# We can overload the main functions to pass the energy range as input
# ----------------------------------------------------------------------

#overload the fitter so that it passes the axisCuts on
class g2FitHistEnergyScan(g2FitHist):
  axisCuts = luigi.Parameter(default=None, description="['axis string', lower bound, upper bound] -> ['y', 1700, 3200]")
  def requires(self):
    return rootToBoost(axisCuts=self.axisCuts) 

class energyScan(law.Task):

    eLow = luigi.FloatParameter(default=400.)
    eHigh = luigi.FloatParameter(default=3150.)
    eWidth = luigi.FloatParameter(default=50.)
    nWorkers = luigi.IntParameter(default=1)

    def requires(self):
        self.elows = np.arange(self.eLow, self.eHigh, self.eWidth)
        for energy in self.elows:
            thiscut = str([['y',energy, energy+self.eWidth]])
            print(thiscut)
            yield g2FitHistEnergyScan(axisCuts=thiscut)

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+GlobalParams().campaignName+"_workers_"+str(self.nWorkers)+"_output_hist_energyScan.pickle")
    
    def run(self):
        input = self.input()
        print("Raw input:", input)

        # nrows = int(np.ceil(len(input)/4))
        # fig,axs = plt.subplots(nrows, 4,figsize=(15,25), sharex=True, sharey=False)
        # if(nrows > 1):
        #     ax = flatten2dArray(axs)
        # else:
        #     ax = axs

        # print(fig,ax)

        fig,ax = plt.subplots(figsize=(8,5))
        NAsquare = []
        NAsquare_err = []
        energies = []

        hists = []
        for i, inputi in enumerate(input):

            elow = self.elows[i]
            ehigh = elow + self.eWidth

            print("Input for the second task:", inputi.path)

            # try:
            opened_hist = pickle.load(open(inputi.path,"rb"))
            histName = list(opened_hist)[0]

            print(opened_hist, histName)

            thisHist = opened_hist[ histName ]
            # except:
            #     print("ERROR: Unable to open file")
            #     continue

            

            print(thisHist)
            print(thisHist.values, thisHist.values[0])

            N = ufloat( thisHist.values[0], thisHist.errors[0] )
            A = ufloat( thisHist.values[2], thisHist.errors[2] )
            NAsquarei = N*A*A
            print(N, A, NAsquarei)

            NAsquare.append(NAsquarei.n)
            NAsquare_err.append(NAsquarei.s)

            energies.append(elow + self.eWidth/2.0)

        plt.errorbar(energies, NAsquare, yerr=NAsquare_err, xerr=self.eWidth,fmt=".-")
        plt.title("NAsqaure Scan")
        plt.xlabel("Energy [MeV]")
        plt.ylabel(r"$NA^{2}$")
        plt.grid()
        plt.tight_layout()
        plt.show()

        data = {'plot':(fig,ax), 'hists':hists, "NAsquare":[energies, NAsquare, NAsquare_err, self.eWidth]}
        self.output().dump(data, formatter="pickle")