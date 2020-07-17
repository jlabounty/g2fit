import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast
import matplotlib.pyplot as plt
import numpy as np

#iMinuit fitting
from  iminuit.cost import LeastSquares
from matplotlib import pyplot as plt
from iminuit import Minuit

# g-2 blinding software
from BlindersPy3 import Blinders
from BlindersPy3 import FitType


class g2Fitter():

    def getBlinded(self,blinding_string = "Test Blinding"):
        # blinding_string = 'This is my fight song. Blinding my plot song.'
        return Blinders(FitType.Omega_a, blinding_string)

    def blinded_5par(self,x,p):
        # print(self, type(self))
        self.parNames = ['N', '#tau', 'A', 'R', '#phi']
        norm  = p[0]
        life  = p[1]
        asym  = p[2]
        R     = p[3]
        phi   = p[4]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        return norm * np.exp(-time/life) * (1 - asym*np.cos(omega*time + phi))
    
    def fit_functions(self, name):
        dicti = {
            "5par":self.blinded_5par
        }
        return dicti[name]

    def cost_functons(self,name):
        dict_cost_functons = {
            "LeastSquares":LeastSquares
        }
        return dict_cost_functons[name]

    def getFunc(self):
        return self.fit_functions(self.fit_name)

    def getCost(self):
        return self.cost_functons(self.cost_name)

    def __init__(self, fit_name, cost_name, blinding_string, boost_hist, initial_guess, 
                       xlims=None, useError=True, verbose=1):
        self.fit_name = fit_name
        self.cost_name = cost_name 
        self.blinding_string = blinding_string
        self.h = boost_hist.copy() #ensure that we can picke things up at the end by creating copy
        self.initial_guess = initial_guess

        if(xlims is not None):
            self.xlims=xlims
            self.binlims = [self.h.axes[0].index(x) for x in xlims]
            print(xlims, "->",self.binlims)
        else:
            self.binlims = [0, len(self.h.axes[0].centers)]

        self.x = self.h.axes[0].centers[self.binlims[0]:self.binlims[1]]
        self.y = self.h.view().value[self.binlims[0]:self.binlims[1]]
        if(useError):
            self.yerr = self.h.view().variance[self.binlims[0]:self.binlims[1]]
        else:
            self.yerr = None 
            self.xerr = None

        # self.fit_function = self.fit_functions[self.fit_name]
        self.fit_function = (self.getFunc())

    def do_fit(self, nFit=2):
        print("Starting fit...")
        cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=1)
        m = Minuit.from_array_func( cost_function, start=self.initial_guess, 
                               name=self.parNames, errordef=1)
        print(m.params)

        for i in range(nFit):
            m.migrad()

        # we cannot pickle the Minuit object directly, so we need to save all of the things we find important!
        # print(m.values)
        self.values = m.np_values()
        self.errors = m.np_errors()
        self.fitarg = m.fitarg
        self.corr = m.np_matrix(correlation=True)
        self.cov = m.np_covariance()

        print(self.fitarg)
        
        print("Fit complete!")

    # get around the fact that we can't save the fit object directly by loading it back up when we need it.
    def load_fit(self, nFit=2):
        print("WARNING: You will not be able to pickle this file")
        cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=1)
        self.m = Minuit.from_array_func(cost_function, self.values, name=self.parNames, **self.fitarg)
        for i in range(nFit):
            self.m.migrad()


    def plot_result(self):
        fig,ax = plt.subplots(figsize=(15,5))
        plt.plot(self.x,self.y)

        self.load_fit()

        plt.plot(self.x, self.fit_function(self.x, self.m.values[:]), color="red"); # trick to access values as a tuple

        plt.yscale("log")
        # plt.xlim(30,40)
        plt.show()
        return(fig,ax)


# test output
infile = "../root_import/data/test2_clustersAndCoincidences_corrected_[['y', 2100, 2200]].pickle"
input_hist = pickle.load(open(infile,"rb"))['clustersAndCoincidences/corrected']

print(type(input_hist))

ding = g2Fitter("5par","LeastSquares","test", input_hist, [1090000,64.4,0.33,0,-1.2], [30,300])
print(ding, type(ding))
print(ding.fit_function)
print(ding.fit_function(np.array([10,11,12]),[10,1,0.33,0,0]))

ding.do_fit()
# ding.plot_result()

pickle.dump(ding,open("./test_fit_output.pickle","wb"), protocol=2)
ding2 = pickle.load(open("./test_fit_output.pickle","rb"))
print(ding2)

ding2.load_fit()
print(ding2.m.values)

print("All done!")