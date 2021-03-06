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
from iminuit.cost import LeastSquares
from iminuit.cost import ExtendedBinnedNLL
from matplotlib import pyplot as plt
from iminuit import Minuit
from iminuit.util import describe, make_func_code


# g-2 blinding software
from BlindersPy3 import Blinders
from BlindersPy3 import FitType


from g2Reader import *


#least squares implementation stolen from 
# https://github.com/scikit-hep/iminuit/blob/master/tutorial/generic_least_squares_function.ipynb
class LeastSquaresBase:
    """
    Generic least-squares cost function.
    """
    def __init__(self, model, x, y, yerr, verbose=0):
        self.model = model  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        if(yerr is None):
            yerr = 1
        else:
            self.yerr = np.array(yerr)
        self.verbose = verbose

    def __call__(self, *par):  # we accept a variable number of model parameters
        ym = self.model(self.x, *par)

        chi2 = np.nansum(((self.y - ym)/self.yerr)**2)
        if(self.verbose > 0):
            print("Chi^2 Statistic for this iteration:", chi2)
        return chi2

class BetterLeastSquares(LeastSquaresBase):
    def __init__(self, model, x, y, yerr):
        super().__init__(model, x, y, yerr, verbose=0)
        self.func_code = make_func_code(describe(model)[1:])


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

        # print(x)
        
        y = norm * np.exp(-time/life) * (1 - asym*np.cos(omega*time + phi))
        # print(y[:10])
        return y

    def fixed_5par(self,x,p):
        '''
            Five parameter g-2 omega_a fit with omega_a fixed to the reference value
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
        # omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        omega = self.getBlinded(self.blinding_string).referenceValue()

        # print(x)
        
        y = norm * np.exp(-time/life) * (1 - asym*np.cos(omega*time + phi))
        # print(y[:10])
        return y

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
        
        cCBO = 1-np.exp(-time/lifeCBO)*A1*np.cos(omegaCBO*np.pi*time + phiCBO1)
        ACBO = asym * (1 - np.exp(-time/lifeCBO) * A2 * np.cos(omegaCBO*np.pi*time + phiCBO2))
        phiCBO = phi + np.exp(-time/lifeCBO)*A3*np.cos(omegaCBO*np.pi*time + phiCBO3)
        
        return norm * np.exp(-time/life) * cCBO * (1 - ACBO*np.cos(omega*time + phiCBO))

    def blinded_17par(self,x,p):
        '''
            17-parameter g-2 omega_a fit, incorporating CBO and VW terms.
            We can set these up such that when masked the parameters can be set to 0
        '''
        npars = 17
        assert len(self.initial_guess) == npars
        self.parNames = ['N', '#tau', 'A', 'R', '#phi', 'A_{CBO - N}', 'A_{CBO - A}', 'A_{CBO - #phi}', 
                         '#tau_{CBO}', '#omega_{CBO}', '#phi_{CBO - N}', '#phi_{CBO - A}', '#phi_{CBO - #phi}',
                         "A_{VW}", "#tau_{VW}", "#omega_{VW}", "#phi_{VW}"]


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
        Avw      = p[13]
        lifeVW   = p[14]
        omegaVW  = p[15]
        phiVW    = p[16]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        cCBO = 1-np.exp(-time/lifeCBO)*A1*np.cos(omegaCBO*np.pi*time + phiCBO1)
        ACBO = asym * (1 - np.exp(-time/lifeCBO) * A2 * np.cos(omegaCBO*np.pi*time + phiCBO2))
        phiCBO = phi + np.exp(-time/lifeCBO)*A3*np.cos(omegaCBO*np.pi*time + phiCBO3)
        cVW = 1 - Avw*np.exp(-time/lifeVW)*np.cos(omegaVW*np.pi*time + phiVW)
        
        return norm * np.exp(-time/life) * cCBO * cVW * (1 - ACBO*np.cos(omega*time + phiCBO))

    def blinded_18par(self,x,p):
        '''
            18-parameter g-2 omega_a fit, incorporating CBO, VW, and Kloss terms.
            We can set these up such that when masked the parameters can be set to 0
        '''
        npars = 18
        assert len(self.initial_guess) == npars
        assert (self.h_Kloss is not None)
        self.parNames = ['N', '#tau', 'A', 'R', '#phi', 
                         'A_{CBO - N}', 'A_{CBO - A}', 'A_{CBO - #phi}', 
                         '#tau_{CBO}', '#omega_{CBO}', '#phi_{CBO - N}', '#phi_{CBO - A}', '#phi_{CBO - #phi}',
                         "A_{VW}", "#tau_{VW}", "#omega_{VW}", "#phi_{VW}", 
                         "K_{loss}"]


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
        Avw      = p[13]
        lifeVW   = p[14]
        omegaVW  = p[15]
        phiVW    = p[16]
        Kloss    = p[17]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        cCBO = 1-np.exp(-time/lifeCBO)*A1*np.cos(omegaCBO*np.pi*time + phiCBO1)
        ACBO = asym * (1 - np.exp(-time/lifeCBO) * A2 * np.cos(omegaCBO*np.pi*time + phiCBO2))
        phiCBO = phi + np.exp(-time/lifeCBO)*A3*np.cos(omegaCBO*np.pi*time + phiCBO3)
        cVW = 1 - Avw*np.exp(-time/lifeVW)*np.cos(omegaVW*np.pi*time + phiVW)
        
        return (
                norm * np.exp(-time/life) * cCBO * cVW * (1 - ACBO*np.cos(omega*time + phiCBO))
                * ( 1 - Kloss*np.interp(time, self.h_Kloss.axes[0].centers, self.h_Kloss.view() ))
               )

    def blinded_21par(self,x,p):
        '''
            21-parameter g-2 omega_a fit, incorporating CBO, VW, 2*CBO, and Kloss terms.
            We can set these up such that when masked the parameters can be set to 0
        '''
        npars = 21
        assert len(self.initial_guess) == npars
        assert (self.h_Kloss is not None)
        self.parNames = ['N', '#tau', 'A', 'R', '#phi', 
                         'A_{CBO - N}', 'A_{CBO - A}', 'A_{CBO - #phi}', 
                         '#tau_{CBO}', '#omega_{CBO}', '#phi_{CBO - N}', '#phi_{CBO - A}', '#phi_{CBO - #phi}',
                         "A_{VW}", "#tau_{VW}", "#omega_{VW}", "#phi_{VW}", 
                         "K_{loss}",
                         "A_{2-CBO}", "#tau_{2-CBO}", "#phi_{2-CBO}"]


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
        Avw      = p[13]
        lifeVW   = p[14]
        omegaVW  = p[15]
        phiVW    = p[16]
        Kloss    = p[17]
        A2CBO    = p[18]
        life2CBO = p[19]
        phi2CBO  = p[20]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        cCBO = 1-np.exp(-time/lifeCBO)*A1*np.cos(omegaCBO*np.pi*time + phiCBO1)
        ACBO = asym * (1 - np.exp(-time/lifeCBO) * A2 * np.cos(omegaCBO*np.pi*time + phiCBO2))
        phiCBO = phi + np.exp(-time/lifeCBO)*A3*np.cos(omegaCBO*np.pi*time + phiCBO3)
        cVW = 1 - Avw*np.exp(-time/lifeVW)*np.cos(omegaVW*np.pi*time + phiVW)
        c2CBO = 1 - A2CBO*np.exp(-time/life2CBO)*np.cos(2*omegaCBO*np.pi*time + phi2CBO)
        
        return (
                norm * np.exp(-time/life) * cCBO * cVW * (1 - ACBO*np.cos(omega*time + phiCBO))
                * ( 1 - Kloss*np.interp(time, self.h_Kloss.axes[0].centers, self.h_Kloss.view() ))
                * c2CBO
               )

    def blinded_25par(self,x,p):
        '''
            25-parameter g-2 omega_a fit, incorporating CBO, VW, 2*CBO, Vy, and Kloss terms.
            We can set these up such that when masked the parameters can be set to 0
        '''
        npars = 25
        assert len(self.initial_guess) == npars
        assert (self.h_Kloss is not None)
        self.parNames = ['N', '#tau', 'A', 'R', '#phi', 
                         'A_{CBO - N}', 'A_{CBO - A}', 'A_{CBO - #phi}', 
                         '#tau_{CBO}', '#omega_{CBO}', '#phi_{CBO - N}', '#phi_{CBO - A}', '#phi_{CBO - #phi}',
                         "A_{VW}", "#tau_{VW}", "#omega_{VW}", "#phi_{VW}", 
                         "K_{loss}",
                         "A_{2-CBO}", "#tau_{2-CBO}", "#phi_{2-CBO}",
                         "A_{y}", "#tau_{y}", "#omega_{y}", "#phi_{y}"]


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
        Avw      = p[13]
        lifeVW   = p[14]
        omegaVW  = p[15]
        phiVW    = p[16]
        Kloss    = p[17]
        A2CBO    = p[18]
        life2CBO = p[19]
        phi2CBO  = p[20]
        Ay       = p[21]
        lifey    = p[22]
        omegay   = p[23]
        phiy     = p[24]
        
        time  = x
        omega = self.getBlinded(self.blinding_string).paramToFreq(R)
        
        cCBO = 1-np.exp(-time/lifeCBO)*A1*np.cos(omegaCBO*np.pi*time + phiCBO1)
        ACBO = asym * (1 - np.exp(-time/lifeCBO) * A2 * np.cos(omegaCBO*np.pi*time + phiCBO2))
        phiCBO = phi + np.exp(-time/lifeCBO)*A3*np.cos(omegaCBO*np.pi*time + phiCBO3)
        cVW = 1 - Avw*np.exp(-time/lifeVW)*np.cos(omegaVW*np.pi*time + phiVW)
        cy = 1 - Ay*np.exp(-time/lifey)*np.cos(omegay*np.pi*time + phiy)
        c2CBO = 1 - A2CBO*np.exp(-time/life2CBO)*np.cos(2*omegaCBO*np.pi*time + phi2CBO)
        
        return (
                norm * np.exp(-time/life) * cCBO * cVW * (1 - ACBO*np.cos(omega*time + phiCBO))
                * ( 1 - Kloss*np.interp(time, self.h_Kloss.axes[0].centers, self.h_Kloss.view() ))
                * c2CBO * cy
               )


    
    #fit functions must be registered here in order for them to be used
    def fit_functions(self, name):
        dicti = {
            "custom":None,
            "5par"  :self.blinded_5par,
            "5parfixed"  :self.fixed_5par,
            "13par" :self.blinded_13par,
            '17par' :self.blinded_17par,
            "18par" :self.blinded_18par,
            "21par" :self.blinded_21par,
            "25par" :self.blinded_25par
        }
        return dicti[name]

    #cost functions must be registered here in order for them to be used
    def cost_functons(self,name):
        dict_cost_functons = {
            "LeastSquares":LeastSquares,
            "NLL":ExtendedBinnedNLL,
            "BLS":BetterLeastSquares
            #TODO: #2 Add custom minimizer which takes advantage of the kernel method
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

    def triples_to_Kloss(self, triples_hist, tau=64.4):
        h = bh.Histogram(*triples_hist.axes)
        # print(h)
        h[:] = ( triples_hist.view()*np.exp( h.axes[0].centers / tau ) )
        h[:] = np.cumsum(h[:].view()) 
        h[:] = h[:].view() / np.sum(h[:].view())
        
        # raise ValueError

        return h

    @staticmethod
    @jit(nopython=True)
    def zip_1d(a,b):
        '''
            We can jit this to make it a bit faster.
            Turns the uproot hist into the correct boost format by combining means/variances into one object
        '''
        output = np.zeros((*a.shape,2))
        for i, a1 in enumerate(a):
            output[i] = [a1, b[i]]
        return output

    def __init__(self, fit_name, cost_name, blinding_string, boost_hist, initial_guess, 
                       xlims=None, useError=False, verbose=1, uniqueName="", do_iterative_fit=False,
                       fit_list=None, fit_limits = None, final_unlimited_fit = False,
                       custom_func=None, parNames=None, triples_hist=None, ignore_zeros=False):
        self.verbose = verbose
        if(self.verbose > 0):
            print("Initializing fit!")

        self.uniqueName = uniqueName
        self.fit_name = fit_name
        self.cost_name = cost_name 
        self.blinding_string = blinding_string
        self.blind = None # to be set up later once we initialize the fit

        #we expect the boost_hist to be 1D
        if(self.verbose > 0):
            print(boost_hist.axes.size, boost_hist.ndim)
        assert (boost_hist.ndim is 1)
        try:
            boost_hist.view().value
            self.h = boost_hist.copy() #ensure that we can pack things up at the end by creating copy
        except:
            # need to convert this histogram into a weighted storage.
            if(self.verbose > 0):
                print("Converting unweighted histogram into expected format")
            a = np.array(boost_hist.view())
            b = np.zeros_like(a)
            self.h = bh.Histogram(*boost_hist.axes, storage=bh.storage.Weight() )
            ding =  self.zip_1d(a,b)
            self.h[:] = ding
        if(triples_hist is not None):
            assert (triples_hist.ndim is 1) #Kloss hist should also be 1D
            self.h_Kloss = self.triples_to_Kloss( triples_hist )
            
        else:
            self.h_Kloss = None
        self.initial_guess = initial_guess
        self.do_iterative_fit = do_iterative_fit
        self.fit_list = fit_list 
        self.intermediate_values = []
        if(fit_limits is not None):
            self.fit_limits = fit_limits
        else:
            self.fit_limits = [None for i in range(len(self.initial_guess))]
        self.final_fit_unlimited = final_unlimited_fit

        self.xlims=xlims
        if(xlims is not None):
            self.binlims = [self.h.axes[0].index(x) for x in xlims]
            self.xlims_true = [self.h.axes[0].bin(x) for x in self.binlims]
            if(self.verbose > 0):
                print(xlims, "->",self.binlims, "(",self.xlims_true,")")
        else:
            self.binlims = [0, len(self.h.axes[0].centers)]

        self.x = self.h.axes[0].centers[self.binlims[0]:self.binlims[1]]
        self.y = self.h.view().value[self.binlims[0]:self.binlims[1]]
        self.useError = useError
        if(useError):
            self.yerr = self.h.view().variance[self.binlims[0]:self.binlims[1]]
        else:
            #needs to be non-zero for least squares minimizer to work
            #default to  sqrt(bin content)
            self.yerr = np.sqrt( self.h.view().value[self.binlims[0]:self.binlims[1]] )  
            self.yerr[self.yerr == 0] = np.nan
        self.xerr = None

        if(ignore_zeros):
            self.y[self.y == 0] = np.nan

        # print(self.x, self.y, self.yerr)

        # self.fit_function = self.fit_functions[self.fit_name]
        if(self.fit_name is not "custom"):
            self.fit_function = (self.getFunc())
        else:
            self.fit_function = custom_func 
            if(parNames is not None):
                self.parNames = parNames
            else:
                self.parNames = ["p"+str(i) for i in range(len(self.initial_guess))]

        #initialize fit function by evaluating once on initial guess
        self.fit_function(np.array([10]),self.initial_guess)
        # print(self)

    def store_fit_values(self, m):
        self.values = m.np_values()
        self.errors = m.np_errors()
        self.fitarg = m.fitarg
        if(self.useError and m.valid):
            self.corr = m.np_matrix(correlation=True)
            self.cov = m.np_covariance()
        else:
            print("Warning: Fit might not be valid, covariance/correlation matrices not stored")
            self.corr = None
            self.cov = None
        self.fity = self.fit_function(self.x, self.values)
        self.chiSquare = m.fval 
        self.NDF = (len(self.y) - len(m.values))
        if(self.NDF is not 0):
            self.reducedChiSquare = self.chiSquare / self.NDF
        else:
            self.reducedChiSquare = np.nan

        if(self.do_iterative_fit):
            self.intermediate_values.append( [self.values, self.errors, self.fitarg, self.chiSquare, self.initial_guess] )
        

    def do_fit(self, nFit=2):
        if(not self.do_iterative_fit):
            # for the single fit, just do a normal migrad call
            self.do_single_fit(nFit)
        else:
            #for the iterative fit, loop over the parameters and unmask them iteratively using the same minuit object
            self.iterative_fit(nFit)

        self.compute_residuals()

    def compute_residuals(self):

        #compute all residuals and pulls
        self.residuals = self.y - self.fity
        self.pulls = np.divide((self.y - self.fity), (self.yerr), out=np.zeros_like(self.y), where=self.yerr!=0)

        #compute the ones which lie in the fit range
        if(self.xlims is not None):
            self.residualsInRange = self.residuals[ self.h.axes[0].index(self.xlims[0]):self.h.axes[0].index(self.xlims[1]) ]
            self.pullsInRange = self.pulls[ self.h.axes[0].index(self.xlims[0]):self.h.axes[0].index(self.xlims[1])  ]
        else:
            self.residualsInRange = self.residuals
            self.pullsInRange = self.pulls

        if(self.verbose > 0):
            print("Residuals computed!")

    def iterative_fit(self, nFit=2):
        if(self.verbose > 0):
            print("Starting iterative fit")

        #create a minuit object
        if(self.verbose > 0):
            print("Cost function:", self.cost_name)
        if(self.cost_name is "LeastSquares"):
            self.cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=0)
        elif(self.cost_name is "NLL"):
            edges = self.h.axes[0].edges[self.binlims[0]:self.binlims[1]+1]
            if(self.verbose > 0):
                print(len(self.y), len(edges))
            self.cost_function = (self.getCost())(self.y, edges, self.fit_function )
        elif(self.cost_name is "BLS"):
            self.cost_function = (self.getCost())(self.fit_function, self.x, self.y, self.yerr)
        else:
            raise ValueError("Undefined cost function:", self.cost_name)
        # self.m = Minuit.from_array_func( self.cost_function, start=self.initial_guess, 
        #                        name=self.parNames, errordef=1)
        self.m = (self.getMinuit())( self.cost_function, start=self.initial_guess, 
                                     name=self.parNames, errordef=1, limit=self.fit_limits)
        
        # print(self.m, self.cost_function)
        if(self.verbose > 0):
            print(self.m.params)
        #loop over the parameters which should be masked + a final unmasked version
        for mask in self.fit_list+[len(self.initial_guess)]:
            if(self.verbose > 0):
                print(mask)
                print('Fitting using these pars:', self.parNames[:mask])
                print("   with values:", self.initial_guess[:mask])

            self.m.fixed[:] = True
            self.m.fixed[:mask] = False

            for i in range(nFit):
                self.m.migrad()
            self.m.hesse()
            self.m.minos()
            self.store_fit_values(self.m)
            if(self.verbose > 0):
                print(self.m.params)

        if(self.final_fit_unlimited):
            #if this option is set, do a final unlimited fit
            # self.m.limits[:] = [None for i in range(len(self.initial_guess))]
            theseargs = self.m.fitarg
            if(self.verbose > 0):
                print(theseargs)
            for key in theseargs:
                if("limit_" in key):
                    theseargs[key] = None
            if(self.verbose > 0):
                print(theseargs)

            #unable to unset limits in minuit??? so have to create a new object...
            self.m = (self.getMinuit())(self.cost_function, start=self.initial_guess, name=self.parNames, **theseargs)

            for i in range(nFit):
                self.m.migrad()
            self.m.hesse()
            self.m.minos()
            self.store_fit_values(self.m)

            if(self.verbose > 0):
                print(self.m.params)

        if(self.verbose > 0):
            print("All done!")

    def do_single_fit(self, nFit=2):
        if(self.verbose > 0):
            print("Starting fit...")

        #TODO: #1 When minuit is pickleable, can transition these to being class functions and ax 'load_fit'
        self.cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=0)
        # self.m = Minuit.from_array_func( self.cost_function, start=self.initial_guess, 
        #                        name=self.parNames, errordef=1)
        self.m = (self.getMinuit())( self.cost_function, start=self.initial_guess, 
                                     name=self.parNames, errordef=1, limit=self.fit_limits)

        if(self.verbose > 0):
            print(self.m, self.cost_function)
            print(self.m.params)

        for i in range(nFit):
            self.m.migrad()
        self.m.hesse()
        self.m.minos()

        if(self.verbose > 0):
            print(self.m.params)

        # we cannot pickle the Minuit object directly, so we need to save all of the things we find important!
        # print(m.values)
        self.store_fit_values(self.m)

        if(self.verbose > 0):
            print(self.fitarg)

        #in order to pickle, need to delete
        # del self.m 
        # del self.cost_function
        
        if(self.verbose > 0):
            print("Fit complete!")

    # get around the fact that we can't save the fit object directly by loading it back up when we need it.
    def load_fit(self, nFit=2):
        if(self.verbose > 0):
            print("WARNING: You will not be able to pickle this file after executing this function")
        cost_function = (self.getCost())(self.x,self.y,self.yerr, self.fit_function, verbose=0)
        self.m = Minuit.from_array_func(cost_function, self.values, name=self.parNames, **self.fitarg)
        for i in range(nFit):
            self.m.migrad()

    def make_pickleable(self):
        del self.m 
        del self.cost_function

        if(self.verbose > 0):
            print(self,"can now be pickled")

    # --------------------------------------------------------
    # Plotting utility functions
    # 

    def plot_result(self):
        fig,ax = plt.subplots(figsize=(15,5))
        plt.plot(self.x,self.y)

        self.load_fit()

        plt.plot(self.x, self.fit_function(self.x, self.m.values[:]), color="red"); # trick to access values as a tuple

        plt.yscale("log")
        # plt.xlim(30,40)
        plt.show()
        return(fig,ax)

    def labelFit(self, parNames=None, functionString=None, formatStr="7.3E", func=None,fitname=None):
        # import ROOT as r
        import textwrap
        import ctypes 

        pars = self.values
        parErrs = self.errors
        chiSquare = self.reducedChiSquare

        if(parNames is None):
            # parNames = ["p"+str(i) for i in range(len(pars))]
            parNames = [convertRootLabelsToPython(str(self.parNames[i])) for i in range(len(pars))]
        else:
            parNames = self.parNames

        if(self.xlims is not None):
            parstring = "Fit Result ["+str(round(self.xlims[0],2))+" - "+str(round(self.xlims[1],2))+"]"+"\n"
        else:
            parstring = "Fit Result"+"\n"

        if(fitname is not None):
            parstring += str(fitname)+"\n"

        if(functionString is not None):
            wrapped = textwrap.wrap(functionString, 25)
            if(len(wrapped) > 0):
                parstring+="Function: "+wrapped[0]+"\n"
            if(len(wrapped)>1):
                for fi in wrapped[1:]:
                    parstring += "                "+fi+"\n"
        for i in range(len(pars)):
            #parstring += str(parNames[i])+"\t= "+str(format(pars[i], formatStr))
            if(parErrs is not None):
                parstring += ("{0}".ljust(10)+" = {1:.3e} $\\pm$ {2:.3e}").format(parNames[i], pars[i], parErrs[i])+"\n"
                #parstring += "\t"+r"$\pm$ "+str(format(parErrs[i], formatStr))+"\n"
            else:
                parstring += ("{0}".ljust(10)+" = {1:.3e}").format(parNames[i], pars[i])+"\n"
                #parstring += "\n"
        if(chiSquare is not None):
            parstring += r"$\chi^{2}/NDF$ = "+str(chiSquare)
            
        # parstring="Placeholder"

        return parstring


    def draw(self,  title = "Fit Result", yrange=[None, None], xrange=None, data_title = "Data", 
                    resid_title = "Fit Pulls", do_pulls=True, fit_hist=True, fmti=".-", 
                    draw_confidence_intervals=False, scaleFactor=None, labelFit=True):
        '''
            Creates a figure in matplotlib and draws the fitresult / residuals on it
        '''

        import matplotlib.pyplot as plt
        import numpy as np

        #fig, axs = plt.subplots(2,1,figsize=(15,8), sharex=True)
        fig = plt.figure(figsize=(15,8))
        gs = fig.add_gridspec(2,3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :-1])
        ax3 = fig.add_subplot(gs[1, -1:])

        axs = [ax1, ax2, ax3]
        ax = axs[0]
        ax.errorbar(self.x, self.y, xerr=self.xerr, yerr=self.yerr, fmt=fmti, 
                    label="Data", ecolor='xkcd:grey', zorder=35)

        if(scaleFactor is None):         
            self.drawFitResult(ax, scaleFactor=int(len(self.x)*10))
        elif(labelFit):
            self.drawFitResult(ax, scaleFactor=scaleFactor)
        else:
            self.drawFitResult(ax, scaleFactor=scaleFactor, label=None)
        #TODO: #3 Re-implement confidence intervals
        # if(draw_confidence_intervals):
        #     self.drawConfidenceIntervals(ax,"blue", "95% Confidence Level")
        ax.grid()
        if(yrange[0] is None):
            ymean = np.nanmean(self.y)
            ystd = np.nanstd(self.y)
            ax.set_ylim(ymean - ystd*2, ymean + ystd*2)
        else:
            ax.set_ylim(yrange[0][0],yrange[0][1])
        if(xrange is not None):
            ax.set_xlim(xrange[0], xrange[1])
        if(labelFit):
            leg = ax.legend(ncol=1, loc=4, fontsize=20*(1-(len(self.values)/28.)))
            leg.set_zorder(37)
        ax.set_ylabel(data_title)

        ax = axs[1]
        if(do_pulls and (self.yerr is not None)):
            ax.set_title("Fit Pulls")
            self.plotResiduals(ax,1,fmti,"All Residuals",1)
        else:
            ax.set_title("Fit Residuals")
            self.plotResiduals(ax,0,fmti,"All Residuals",1)
        ax.set_ylabel(resid_title)
        ax.grid()
        if(yrange[1] is None):
            ymean = np.nanmean(self.residuals)
            ystd = np.nanstd(self.residuals)
            ax.set_ylim(ymean - ystd*2, ymean + ystd*2)
        else:
            ax.set_ylim(yrange[1][0],yrange[1][1])
        #ax.set_ylim(-10,10)
        if(xrange is not None):
            ax.set_xlim(xrange[0], xrange[1])
        
        ax = axs[2]
        ax.grid()
            
        if(do_pulls and (self.yerr is not None)):
            ax.set_title("Histogram of Fit Pulls")
            hist1 = ax.hist(self.pullsInRange, bins=41)
        elif(np.sum(self.residualsInRange) is not np.nan):
            ax.set_title("Histogram of Fit Residuals")
            hist1 = ax.hist(self.residualsInRange, bins=41)

            
        # if(fit_hist):
        #     fitHistGaussian(hist1, ax, True, True)
        #     ax.legend()

        # plt.suptitle(title, y=1.0, fontsize=18)
        plt.tight_layout()
        
        return (fig, axs)

    def drawFitResult(self, ax, scaleFactor=100, drawFormat="-", label="Default", 
                      color="xkcd:orange", fitname=None, xrange=None):
        #print(fitresult)
        '''
            Draws a fit result object returned by fitVector function on 'ax' (axes or plot from matplotlib)
        '''
        import numpy as np
        #from standardInclude import labelFit

        if(scaleFactor == 1):
            xs = self.x
            ys = self.fity
        elif(xrange is not None):
            x0 = xrange[0]
            xn = xrange[1]
            xs = np.linspace(x0,xn,scaleFactor)
            ys = self.fit_function(xs, self.values)
        elif(scaleFactor > 1):
            x0 = self.x[0]
            xn = self.x[len(self.x)-1]
            xs = np.linspace(x0,xn,scaleFactor)
            ys = self.fit_function(xs, self.values)
        else:
            xs = self.x
            ys = self.fity
        
        if(label == "Default"):
            labelString = self.labelFit( None, "Test", "7.3E", None, fitname=None) 
        else:
            labelString = label
        
        ax.plot(xs,ys,drawFormat,label=labelString, zorder=36, color=color)
        

    def plotResiduals(self, ax, to_plot = 0, fmt=".",labeli="Fit Residuals", runningAverage = 1):
        '''
            Inputs:
                ax             - axis on which to plot
                to_plot        - 0 = residuals; 1 = pulls
                format         - string of formats
                labeli         - what to put in the legend
                runningAverage - Number of points around the residual to compute the running average for
            Outputs:
                matplotlib object containing residuals            
        '''
        import numpy as np

        x = self.x
        if(to_plot == 1):
            resid = self.pulls
        else:
            resid = self.residuals
        
        resid_to_plot = []
        if(runningAverage > 1):
            #resid = [np.nan for i in range(runningAverage)]+resid+[np.nan for i in range(runningAverage)]
            for i in range(len(x)):
                #ri = resid[i]
                if((i >= runningAverage) and ((len(x) - i) >= runningAverage)):
                    mean_ri = np.nanmean(resid[i-runningAverage:i+runningAverage])
                else:
                    mean_ri = np.nan
                resid_to_plot.append(mean_ri)
        else:
            resid_to_plot = resid
        
        #print(resid_to_plot)
        ding = ax.plot(x,resid_to_plot,fmt,label=labeli)
        return ding

    def plotRunningAverage(self, averages=[1,10,100], showplot=True, fmti="."):
        import matplotlib.pyplot as plt  
        fig,ax = plt.subplots(figsize=(15,5))
        for average in averages:
            self.plotResiduals(ax, 1, fmti, str(average), average)
        plt.grid()
        plt.title("Running Average of Residuals")
        plt.legend()
        return (fig,ax)

    def fft(self, option=1, xrange=None,  
                  time_conversion_factor = 10**(-6), drawHist = True, logy=True):
        '''
            Performs an FFT using python on the functions stored in this class. 
            Options:
                0 = Input function
                1 = Residuals
                2 = Pulls
            xrange: Range over which to perform the FFT. Can be used to exclude data points at beginning/end.
            time_conversion_factor: conversion factor to get the x axis time units to seconds
            drawHist: whether or not to draw the histogram
            logy : Should the fft axis be a log scale
        '''
        import numpy as np
        import matplotlib.pyplot as plt

        if(option == 0):
            points_raw = self.y
        elif(option == 1):
            points_raw = self.residuals
        elif(option == 2):
            points_raw = self.pulls
        else:
            raise ValueError("Invalid option in fft: "+str(option))

        if(xrange is None):
            xrange = self.xlims

        # if(xrange is not None):
        points = []
        xs = []
        print("Restricting range of FFT to:", xrange)
        if(len(xrange) is not 2):
            raise ValueError("xrange is not well defined")
        for i, xi in enumerate(self.x):
            if(xi >= xrange[0] and xi <= xrange[1]):
                points.append(points_raw[i])
                xs.append(xi)
        # else:
        #     xs = self.x
        #     points = points_raw

        ding = np.fft.fft(np.array(points))
        n = len(points)
        d = xs[1] - xs[0]
        freq = np.fft.fftfreq(n, d)
        if(drawHist):
            fig, ax = plt.subplots(1,2,figsize=(15, 5))
            for axi in ax:
                axi.grid()
            ax[0].plot(xs,points,"-")
            ax[0].set_title("Input function")
            ax[1].set_title("FFT result")
            ax[1].plot([np.abs(x) for x in freq],np.abs(ding),'-',label='FFT Result')
            omega_a_ref = 0.2291 #MHZ
            plt.plot([omega_a_ref, omega_a_ref],[0,10**10], ":", label=r"$\omega_{a} = 0.229$")
            for i, name in enumerate(self.parNames):
                if("mega" in name):
                    plt.plot([self.values[i], self.values[i]],[0,10**10], ":", 
                             label=convertRootLabelsToPython(name)+f"$ = {round(self.values[i],3)}$")
            #ax[1].set_xlim(0,50)
            if(logy):
                ax[1].set_yscale("log")
            plt.xlabel("Frequency [MHz]")
            ax[1].set_ylim(np.min( np.abs(ding) ), 1.5*np.max( np.abs(ding) ))
            #plt.xlim(0,6.7157787731503555 / 2)# *10.**6)
            plt.legend()
            # plt.suptitle("FFT Of Fit Result", y=1.02, fontsize=18)
            plt.tight_layout()
            plt.show()

        return (freq, ding)


def convertRootLabelsToPython(label):
        '''
            ROOT and Python parse LaTeX slightly differently for their legends. Lets convert between them.
        '''
        if("#" in label or "_" in label):
            #print(label)

            ding = label.replace("#","\\")
            ding = r"$"+ding+"$ "

            #print(ding)
            return ding
        else:
            return label
