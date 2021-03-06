import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast
# import matplotlib.pyplot as plt

from root_to_boost import rootToBoost
from global_params import GlobalParams

from g2Fitter import *
from rocks_submission import *
from rocks_runner import *
from rocks_runner import _smart_copy



class g2FitHist(SGEJobTask):
    '''
        Takes an output from root_to_boost and fits it in the luigi pipeline
    '''

    # sandbox = "bash::fit_env.sh"

    xlims = luigi.Parameter(default=None)
    # parNames = luigi.Parameter()
    whichFit = luigi.Parameter()
    whichCost = luigi.Parameter()
    initialGuess = luigi.Parameter()
    blindingString = luigi.Parameter()
    parameterLimits = luigi.Parameter(default=None)
    outDir = luigi.Parameter()

    def requires(self):
        # for elow in range(500,2300,100):
        #     #using yield means that this can be parallelized across N workers 
        #     #   ('--workers N'  on command line)
        #     yield rootToBoost(axisCuts="[['y', "+str(elow)+", "+str(elow+100)+"]]")
        return rootToBoost(outDir=self.outDir)

    def output(self):
        return law.LocalFileTarget(f"{self.outDir}/_worker_{str(self.task_id)}_output_fit.pickle")

    def work(self):
        input = self.input()
        print("Raw input:", input)

        #open the histogram object
        working_dir = self._get_working_dir()
        os.chdir(working_dir)
        print("Current directory:", os.curdir )
        
        if(self.run_locally):
            opened_hist = pickle.load(open(input.path,"rb"))
        else:
            _smart_copy(input.path,working_dir)
            opened_hist = pickle.load(open(input.path.split("/")[-1],"rb"))
        histName = list(opened_hist)[0]
        print(opened_hist, histName)
        thisHist = opened_hist[ histName ]

        #parse some configuration args which appear as lists
        # fit = g2Fitter("5par","LeastSquares","test", input_hist, [1090000,64.4,0.33,0,-1.2], [30,300], )
        xlims_parsed = [float(x) for x in ast.literal_eval(self.xlims)]
        # parNames_parsed = ast.literal_eval(self.parNames)
        parGuess_parsed = [float(x) for x in ast.literal_eval(self.initialGuess)]

        parLimits_parsed = ast.literal_eval(self.parameterLimits)
        
        print(xlims_parsed)
        print(parGuess_parsed)
        print(parLimits_parsed)

        #create the fitter
        fit = g2Fitter(self.whichFit, self.whichCost, self.blindingString, thisHist, 
                       initial_guess=parGuess_parsed, xlims=xlims_parsed, uniqueName=str(self.task_id),
                       fit_limits=parLimits_parsed )
        print("This fit:", fit)
        fit.do_fit() #execute the minimization

        # fit.m.draw_mncontour("A","R")
        # plt.show()

        fit.make_pickleable() #delete the objects which can't be stored in pickle format
        print(fit.fitarg)

        data = {histName:fit}
        if(self.run_locally):
            self.output().dump(data, formatter="pickle")
        else:
            self._smart_dump(data, formatter="pickle")
        # self.output().dump(data, formatter="pickle")
        
        print("Fit complete!")


#overload the fitter so that it passes the axisCuts on
class g2FitHistAxisCuts(g2FitHist):
  axisCuts = luigi.Parameter(default=None, description="['axis string', lower bound, upper bound] -> ['y', 1700, 3200]")
  def requires(self):
    return rootToBoost(axisCuts=self.axisCuts, outDir=self.outDir, dont_remove_tmp_dir=self.dont_remove_tmp_dir) 