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

class energyScanNAsquare(law.Task):

    eLow = luigi.FloatParameter(default=400.)
    eHigh = luigi.FloatParameter(default=3150.)
    eWidth = luigi.FloatParameter(default=50.)
    nWorkers = luigi.IntParameter(default=1)

    def requires(self):
        self.elows = np.arange(self.eLow, self.eHigh, self.eWidth)
        for energy in self.elows:
            thiscut = str([['y',energy, energy+self.eWidth]])
            print(thiscut)
            yield g2FitHistAxisCuts(axisCuts=thiscut)

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

        # fig,ax = plt.subplots(figsize=(8,5))
        NAsquare = []
        NAsquare_err = []
        energies = []
        precision = []
        rvals = []

        hists = []

        r_precision_hist = bh.Histogram(bh.axis.Regular(len(self.elows), self.eLow, self.eHigh), storage=bh.storage.Weight())
        for i, inputi in enumerate(input):

            elow = self.elows[i]
            ehigh = elow + self.eWidth

            print("Input for the second task:", inputi.path)

            try:
                opened_hist = pickle.load(open(inputi.path,"rb"))
            except:
                print("Warning: unable to open", inputi)
                continue

            histName = list(opened_hist)[0]

            print(opened_hist, histName)

            thisHist = opened_hist[ histName ]

            print(thisHist)
            print(thisHist.values, thisHist.values[0])

            N = ufloat( thisHist.values[0], thisHist.errors[0] )
            A = ufloat( thisHist.values[2], thisHist.errors[2] )
            NAsquarei = N*A*A
            print(N, A, NAsquarei)

            precision.append(thisHist.errors[3])
            rvals.append(thisHist.values[3])
            r_precision_hist.fill(elow, weight=thisHist.errors[3])

            NAsquare.append(NAsquarei.n)
            NAsquare_err.append(NAsquarei.s)

            energies.append(elow + self.eWidth/2.0)

        # plt.errorbar(energies, NAsquare, yerr=NAsquare_err, xerr=self.eWidth,fmt=".-")
        # plt.title("NAsqaure Scan")
        # plt.xlabel("Energy [MeV]")
        # plt.ylabel(r"$NA^{2}$")
        # plt.grid()
        # plt.tight_layout()
        # plt.show()

        print(r_precision_hist)
        print(r_precision_hist.axes[0].centers)
        print(r_precision_hist[:])


        fig,ax = plt.subplots(1,2,figsize=(15,5))
        plt.suptitle(r"T-Method Threshold $\chi^2$ Scan",y=1.02,fontsize=18)
        
        plt.sca(ax[0])
        plt.ylim(rvals[0]-2,rvals[0]+2)
        plt.errorbar(self.elows, rvals, yerr=precision, xerr=None,fmt=".-")
        plt.ylabel(r"$R \pm \delta R $ [ppm]")
        plt.xlabel("Energy [MeV]")
        plt.grid()

        plt.sca(ax[1])
        plt.ylim(0,2)
        # plt.errorbar(self.elows, precision, yerr=None, xerr=None,fmt=".-")
        plt.errorbar(r_precision_hist.axes[0].centers[:], r_precision_hist.view().value , yerr=None, xerr=None,fmt=".-")

        def pol2(x,p):
            return p[0] + p[1]*x + p[2]*x*x

        fit = g2Fitter("custom", "LeastSquares", "", r_precision_hist, [0.5,1,1], [1400,2500], 
                        custom_func=pol2, useError=True )
        print("Fit", fit)
        fit.do_fit()

        p0 = ufloat(fit.values[0],fit.errors[0])
        p1 = ufloat(fit.values[1],fit.errors[1])
        p2 = ufloat(fit.values[2],fit.errors[2])

        plt.plot(fit.x, fit.fity,label="Fit Minimum at: {:10.4f}".format(-1*p1/(2*p2)))
        plt.legend()

        plt.xlabel("Energy [MeV]")
        plt.ylabel(r"$\delta R $ [ppm]")
        plt.grid()

        plt.tight_layout()
        # plt.show()

        data = {'plot':(fig,ax), 'hists':hists, 
                "NAsquare":[energies, NAsquare, NAsquare_err, self.eWidth], 
                "chiSquare":[precision]}
        self.output().dump(data, formatter="pickle")

class energyScanTMethodThreshold(energyScanNAsquare):
    def requires(self):
        self.elows = np.arange(self.eLow, self.eHigh, self.eWidth)
        for energy in self.elows:
            thiscut = str([['y',energy, self.eHigh]])
            print(thiscut)
            yield g2FitHistEnergyScan(axisCuts=thiscut)

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+GlobalParams().campaignName+"_workers_"+str(self.nWorkers)+"_output_hist_energyScan_TMethod.pickle")
    