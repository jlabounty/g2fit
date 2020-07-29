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
from g2_fit import g2FitHistAxisCuts


class startTimeScan(law.Task):

    tLow = luigi.FloatParameter(default=30.)
    tHigh = luigi.FloatParameter(default=450.)
    tWidth = luigi.FloatParameter(default=1.)

    tEnd = luigi.FloatParameter(default=650.)

    eLow = luigi.FloatParameter(default=1700.)
    eHigh = luigi.FloatParameter(default=3200.)

    caloNum = luigi.IntParameter(default=0)

    def requires(self):
        self.tlows = np.arange(self.tLow, self.tHigh, self.tWidth)
        if(self.caloNum < 1):
            thiscut = str([['y',self.eLow, self.eHigh]])
        else:
            thiscut = str([['z', self.caloNum-1, self.caloNum-1],['y',self.eLow, self.eHigh]])
        for time in self.tlows:
            xlims=str([time,self.tEnd])
            yield g2FitHistAxisCuts( axisCuts=thiscut, xlims=xlims )

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+GlobalParams().campaignName+
                                    "_StartTimeScan_"+str(self.tLow)+"_"+str(self.tHigh)+"_"+str(self.tWidth)+"_"+
                                    "calo_"+str(self.caloNum)+"_"+
                                    "_output_hist_startTimeScan.pickle")

    def run(self):
        input = self.input()
        print("Raw input:", input)
    
        fit_results = []
        paramScans = {}
        paramScanErrs = {}
        parNames = []
        startTimes = []
        for i, inputi in enumerate(input):
            print("Input for the second task:", inputi.path)
            opened_hist = pickle.load(open(inputi.path,"rb"))
            
            histName = list(opened_hist)[0]

            print(opened_hist, histName)

            thisHist = opened_hist[ histName ]

            fit_results.append(thisHist)
            # print(thisHist.values)
            # print(thisHist.errors)
            # print(thisHist.parNames)
            

            for j, param in enumerate(thisHist.parNames):
                print(param)
                if(i < 1):
                    parNames.append(param)
                    paramScans[param] = []
                    paramScanErrs[param] = []
                paramScans[param].append(thisHist.values[j])
                paramScanErrs[param].append(thisHist.errors[j])
            startTimes.append(thisHist.xlims_true[0][0])

        fig,ax = plt.subplots(len(parNames),1,figsize=(15,6), sharex=True)
        for i,axi in enumerate(ax):
            plt.sca(axi)
            thispar = parNames[i]
            # print(len(startTimes))
            # print(len(paramScans[thispar]))
            plt.errorbar(startTimes, paramScans[thispar], yerr=paramScanErrs[thispar], fmt=".:" )
            # print(paramScanErrs[thispar])
            
            #kawall band as in Eq. 6.16 of Aarons thesis
            kawall_band = [np.sqrt( x**2 - paramScanErrs[thispar][0]**2  ) for x in paramScanErrs[thispar]]
            # print(kawall_band)
            plt.fill_between(startTimes, 
                         [paramScans[thispar][0] + kawall_band[i] for i,x in enumerate(startTimes)], 
                         [paramScans[thispar][0] - kawall_band[i] for i,x in enumerate(startTimes)],
                         alpha = 0.3, interpolate=True, color='red')
            plt.title(thispar)
            plt.grid()
        plt.tight_layout()
        # plt.show()


        data = {
            'fit_results':fit_results, 
            "plot":(fig,ax), 
            "startTimeScan":[parNames, startTimes, paramScans, paramScanErrs], 
            "caloNum":self.caloNum, 
            'tlows':self.tlows,
            'tWidth':self.tWidth,
            'tEnd':self.tEnd,
            'eLow':self.eLow,
            'eHigh':self.eHigh
        }
        self.output().dump(data, formatter="pickle")


class startTimeScanByCalo(law.Task):
    def requires(self):
        for calo in range(1,25):
            yield startTimeScan(caloNum=calo)

    def output(self):
        return law.LocalFileTarget(GlobalParams().outputDir+GlobalParams().campaignName+
                                    "_worker_"+str(self.task_id)+"_"+
                                    "_output_hist_startTimeScanByCalo.pickle")

    def run(self):
        input = self.input()
        print("Raw input:", input)
    
        fit_results = {}
        for i, inputi in enumerate(input):
            print("Input for the second task:", inputi.path)
            f = pickle.load(open(inputi.path,"rb"))
            
            print(f.keys())

            fit_results[f['caloNum']] = f['startTimeScan']

        output = {
            "startTimeScan":fit_results
        }

        self.output().dump(output, formatter="pickle")
