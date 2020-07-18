import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast
# import matplotlib.pyplot as plt
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
        '''
            Function which interfaces with the g-2 blinding software. 
            Checks if blinding function is defined and (if necessary) creates/returns it
        '''
        # blinding_string = 'This is my fight song. Blinding my plot song.'
        if(self.blind is None):
            self.blind =  Blinders(FitType.Omega_a, blinding_string)
        return self.blind

    def blinded_5par(self,x,p):
        '''
            Five parameter g-2 omega_a fit
        '''
        # print(self, type(self))
        npars = 5
        assert len(self.initial_guess) == npars
        self.parNames = ['N', '#tau', 'A', 'R', '#phi']
        norm  = p[0]
        life  = p[1]
        asym  = p[2]
        R     = p[3]
        phi   = p[4]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        return norm * np.exp(-time/life) * (1 - asym*np.cos(omega*time + phi))

    def blinded_13par(self,x,p):
        '''
            13-parameter g-2 omega_a fit, incorporating CBO terms.
            We can set these up such that when masked the parameters can be set to 0
        '''
        npars = 13
        assert len(self.initial_guess) == npars
        self.parNames = ['N', '#tau', 'A', 'R', '#phi', 'A_{CBO - N}', 'A_{CBO - A}', 'A_{CBO - #phi}', 
                         '#tau_{CBO}', '#omega_{CBO}', '#phi_{CBO - N}', '#phi_{CBO - A}', '#phi_{CBO - #phi}']


        norm     = p[0]
        life     = p[1]
        asym     = p[2]
        R        = p[3]
        phi      = p[4]
        A1       = p[5]
        A2       = p[6]
        A3       = p[7]
        lifeCBO  = p[8]
        omegaCBO = p[9]
        phiCBO1  = p[10]
        phiCBO2  = p[11]
        phiCBO3  = p[12]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        cCBO = 1-np.exp(-time/lifeCBO)*A1*np.cos(omegaCBO*time + phiCBO1)
        ACBO = asym * (1 - np.exp(-time/lifeCBO) * A2 * np.cos(omegaCBO*time + phiCBO2))
        phiCBO = phi + np.exp(-time/lifeCBO)*A3*np.cos(omegaCBO*time + phiCBO3)
        
        return norm * np.exp(-time/life) * cCBO * (1 - ACBO*np.cos(omega*time + phiCBO))

    
    #fit functions must be registered here in order for them to be used
    def fit_functions(self, name):
        dicti = {
            "custom":None,
            "5par":self.blinded_5par,
            "13par":self.blinded_13par
        }
        return dicti[name]

    #cost functions must be registered here in order for them to be used
    def cost_functons(self,name):
        dict_cost_functons = {
            "LeastSquares":LeastSquares
        }
        return dict_cost_functons[name]

    def minuit(self, name):
        dict_minuit = {
            "minuit_array":Minuit.from_array_func
        }
        return dict_minuit[name]

    def getFunc(self):
        return self.fit_functions(self.fit_name)

    def getCost(self):
        return self.cost_functons(self.cost_name)

    def getMinuit(self):
        return self.minuit('minuit_array')

    def __init__(self, fit_name, cost_name, blinding_string, boost_hist, initial_guess, 
                       xlims=None, useError=True, verbose=1, uniqueName="", do_iterative_fit=False,
                       fit_list=None, fit_limits = None, final_unlimited_fit = False,
                       custom_func=None, parNames=None):
        self.uniqueName = uniqueName
        self.blind = None
        self.fit_name = fit_name
        self.cost_name = cost_name 
        self.blinding_string = blinding_string
        self.h = boost_hist.copy() #ensure that we can pack things up at the end by creating copy
        self.initial_guess = initial_guess
        self.do_iterative_fit = do_iterative_fit
        self.fit_list = fit_list 
        self.intermediate_values = []
        if(fit_limits is not None):
            self.fit_limits = fit_limits
        else:
            self.fit_limits = [None for i in range(len(self.initial_guess))]
        self.final_fit_unlimited = final_unlimited_fit

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
        if(self.fit_name is not "custom"):
            self.fit_function = (self.getFunc())
        else:
            self.fit_function = custom_func 
            if(parNames is not None):
                self.parNames = parNames
            else:
                self.parNames = ["x"+str(i) for i in range(len(self.initial_guess))]
        #initialize fit function by evaluating once on initial guess
        print(self.fit_function(np.array([10]),self.initial_guess))
        print(self)

    def store_fit_values(self, m):
        self.values = m.np_values()
        self.errors = m.np_errors()
        self.fitarg = m.fitarg
        self.corr = m.np_matrix(correlation=True)
        self.cov = m.np_covariance()
        self.fity = self.fit_function(self.x, self.values)
        self.chiSquare = self.cost_function(self.values)

        if(self.do_iterative_fit):
            self.intermediate_values.append( [self.values, self.errors, self.fitarg, self.chiSquare, self.initial_guess] )
        

    def do_fit(self, nFit=2):
        if(not self.do_iterative_fit):
            # for the single fit, just do a normal migrad call
            self.do_single_fit(nFit)
        else:
            #for the iterative fit, loop over the parameters and unmask them iteratively using the same minuit object
            self.iterative_fit(nFit)

    def iterative_fit(self, nFit=2):
        print("Starting iterative fit")

        #create a minuit object
        self.cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=0)
        # self.m = Minuit.from_array_func( self.cost_function, start=self.initial_guess, 
        #                        name=self.parNames, errordef=1)
        self.m = (self.getMinuit())( self.cost_function, start=self.initial_guess, 
                                     name=self.parNames, errordef=1, limit=self.fit_limits)
        
        print(self.m, self.cost_function)
        print(self.m.params)
        #loop over the parameters which should be masked + a final unmasked version
        for mask in self.fit_list+[len(self.initial_guess)]:
            print(mask)
            print('Fitting using these pars:', self.parNames[:mask])
            print("   with values:", self.initial_guess[:mask])

            self.m.fixed[:] = True
            self.m.fixed[:mask] = False

            for i in range(nFit):
                self.m.migrad()
            self.m.hesse()
            self.store_fit_values(self.m)

            print(self.m.params)

        if(self.final_fit_unlimited):
            #if this option is set, do a final unlimited fit
            # self.m.limits[:] = [None for i in range(len(self.initial_guess))]
            theseargs = self.m.fitarg
            print(theseargs)
            for key in theseargs:
                if("limit_" in key):
                    theseargs[key] = None
            print(theseargs)

            #unable to unset limits in minuit??? so have to create a new object...
            self.m = (self.getMinuit())(self.cost_function, start=self.initial_guess, name=self.parNames, **theseargs)

            for i in range(nFit):
                self.m.migrad()
            self.m.hesse()
            self.store_fit_values(self.m)

            print(self.m.params)

        print("All done!")

    def do_single_fit(self, nFit=2):
        print("Starting fit...")

        #TODO: #1 When minuit is pickleable, can transition these to being class functions and ax 'load_fit'
        self.cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=0)
        # self.m = Minuit.from_array_func( self.cost_function, start=self.initial_guess, 
        #                        name=self.parNames, errordef=1)
        self.m = (self.getMinuit())( self.cost_function, start=self.initial_guess, 
                                     name=self.parNames, errordef=1)

        
        print(self.m, self.cost_function)
        print(self.m.params)

        for i in range(nFit):
            self.m.migrad()
        self.m.hesse()

        print(self.m.params)

        # we cannot pickle the Minuit object directly, so we need to save all of the things we find important!
        # print(m.values)
        self.store_fit_values(self.m)

        print(self.fitarg)

        #in order to pickle, need to delete
        # del self.m 
        # del self.cost_function
        
        print("Fit complete!")

    # get around the fact that we can't save the fit object directly by loading it back up when we need it.
    def load_fit(self, nFit=2):
        print("WARNING: You will not be able to pickle this file after executing this function")
        cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=0)
        self.m = Minuit.from_array_func(cost_function, self.values, name=self.parNames, **self.fitarg)
        for i in range(nFit):
            self.m.migrad()

    def make_pickleale(self):
        del self.m 
        del self.cost_function

        print(self,"can now be pickled")


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
# infile = "../root_import/data/test2_clustersAndCoincidences_corrected_[['y', 2100, 2200]].pickle"
# input_hist = pickle.load(open(infile,"rb"))['clustersAndCoincidences/corrected']

# print(type(input_hist))

# ding = g2Fitter("5par","LeastSquares","test", input_hist, [1090000,64.4,0.33,0,-1.2], [30,300])
# print(ding, type(ding))
# print(ding.fit_function)
# print(ding.fit_function(np.array([10,11,12]),[10,1,0.33,0,0]))

# ding.do_fit()
# # ding.plot_result()

# pickle.dump(ding,open("./test_fit_output.pickle","wb"), protocol=2)
# ding2 = pickle.load(open("./test_fit_output.pickle","rb"))
# print(ding2)

# ding2.load_fit()
# print(ding2.m.values)

# print("All done!")