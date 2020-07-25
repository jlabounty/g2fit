import uproot
import boost_histogram as bh
import numba
from numba import jit
import numpy as np
import ROOT as r


check_hist_cpp = '''
    int checkHist(std::size_t size, const TH3* hist) 
    {
        if (hist == nullptr) 
        {
            return -1;
        }

        std::size_t nHistBins =
        hist->GetNbinsX() * hist->GetNbinsY() * hist->GetNbinsZ();
        if (nHistBins != size) 
        {
            return -2;
        }

        return 0;
    }

'''
r.gInterpreter.ProcessLine(check_hist_cpp)


cpp_code = '''
    int hist3dToNumpyArray(double* data, double* error, std::size_t size, const char* histName)     {
        const TH3* hist = (TH3*)gROOT->FindObject(histName);
        int retcode = checkHist(size, hist);
        if (retcode != 0) 
        {
            return retcode;
        }

        for (int xBin = 1; xBin <= hist->GetNbinsX(); ++xBin) 
        {
            for (int yBin = 1; yBin <= hist->GetNbinsY(); ++yBin) 
            {
                for (int zBin = 1; zBin <= hist->GetNbinsZ(); ++zBin) 
                {
                    *data++ = hist->GetBinContent(xBin, yBin, zBin);
                    *error++ = hist->GetBinError(xBin, yBin, zBin);
                }
            }
        }

        return 0;
    }
'''
r.gInterpreter.ProcessLine(cpp_code)


class g2Histogram():
    '''
        Reads in a root histogram into the boost format using uproot. 
        Uses the file type to determine how to read this in
    '''
    def __init__(self, infile, histname=None, kLossName = None):
        
        if(".root" in infile and (histname is not None)):
            self.read_from_root(infile, histname, kLossName)
        elif(".pickle" in infile):
            return self.read_from_pickle(infile)
        else:
            raise ValueError("ERROR: Unsupported input type (must have .root or .pickle suffix)")

    def read_from_pickle(self, infile):
        import pickle
        try:
            pickled = pickle.load(open(infile,"rb"))
        except:
            raise FileNotFoundError("Unable to open/unpickle file:", infile)

        # self = pickled
        # print(self)
        # print(self.h)
        # self.h = pickled.h.copy()

        # del pickled
        return pickled

    def save(self, outfile=None):
        import time
        import pickle

        if(outfile is None):
            outfile = "g2Histogram_"+str(time.gmtime())+".pickle"
        print("Saving to:", outfile)

        pickle.dump(self,open(outfile,"wb"))
        print("All done!")
        return outfile 


    def read_from_root(self, infile, histname, kLossName = None):
        print("Reading in hist "+str(histname)+" from root file "+str(infile))
        import uproot

        #Following ROOT FILE FORMAT EXAMPLE in boost-histogram docs: 
        #   https://readthedocs.org/projects/boost-histogram/downloads/pdf/latest/

        #use uproot to get the parameters in an easy way
        root_file = uproot.open(infile)
        uproot_hist = root_file[histname]
        print(uproot_hist.__dict__)
        # print(type(uproot_hist.edges))
        # self.uproot_hist = uproot_hist

        self.ndim = len(uproot_hist.edges)

        boost_axes = []
        for axis in uproot_hist.edges:
            boost_axes.append( bh.axis.Variable(axis) )
        # uproot_hist.show()

        self.h = bh.Histogram(*boost_axes, storage=bh.storage.Weight() )

        # error_array = np.zeros( self.f.view(flow=True).shape )
        # self.fill_error_array(error_array)
        self.h_doublePileup = bh.Histogram( *boost_axes ) 

        if(kLossName is not None):
            print("Importing Muon Loss Histogram:", kLossName)
            uproot_hist_Kloss = root_file[kLossName]
            # assert uproot_hist_Kloss.axes.size == 2
            self.h_Kloss = bh.Histogram( bh.axis.Variable(uproot_hist_Kloss.edges[0]),
                                         bh.axis.Variable(uproot_hist_Kloss.edges[1]) ) #2D histogram of calo vs. triple
            self.h_Kloss[:,:] = uproot_hist_Kloss.allvalues #don't care about bin errors here, so uproot is fine
        else:
            self.h_Kloss = None 

        # print("starting zip")

        #use root for the actual io, because uproot handles bin errors in a weird way
        f = r.TFile(infile)
        root_hist = f.Get(histname).Clone()
        print("starting zip")
        # zipped = self.zip(uproot_hist.allvalues, uproot_hist.allvariances )
        # zipped = self.zip( *self.rootHistToNumpy(root_hist) )
        self.h[:,:,:] = self.zip( *self.rootHistToNumpy(root_hist) )
        print("All done!")

    def project(self,dimension=0, xlims=None, ylims=None, zlims=None):
        '''
            Implements projection of the contained boost-histogram object with x/y/z limits
        '''
        if(type(dimension) is str):
            # translate dimensions strings into numbers
            dimdict = {"x":0, "y":1, "z":2, "xy":(0,1), "xz":(0,2), "yz":(1,2)}
            try:
                dimension = dimdict[dimension]
            except:
                raise ValueError("ERROR: Dimensional arrangement not found in", dimdict)
        
        #translate limits into cuts on the histogram
        theselims = []
        for i,lim in enumerate([xlims, ylims, zlims]):
            if lim is not None:
                theselims.append( [self.h.axes[i].index(lim[0]), self.h.axes[i].index(lim[1])+1] )
            else:
                theselims.append( [0, self.h.axes[i].size])
        print("Projecting with:", [xlims, ylims,zlims],"->",theselims)

        #make projection, currently 3D only
        hi = self.h[bh.loc(theselims[0][0]):bh.loc(theselims[0][1]), 
                    bh.loc(theselims[1][0]):bh.loc(theselims[1][1]), 
                    bh.loc(theselims[2][0]):bh.loc(theselims[2][1]), ].project(dimension)
        return hi

    
    def zip(self, a, b):
        zipdict = {
            1:self.zip_1d,
            2:self.zip_2d,
            3:self.zip_3d
        }
        return zipdict[self.ndim](a,b)

    @staticmethod
    @jit(nopython=True)
    def zip_1d(a,b):
        '''
            We can jit this to make it a bit faster.
            Turns the uproot hist into the correct boost format by combining means/variances into one object
        '''
        output = np.zeros((*a.shape,2))
        for i, a1 in enumerate(a):
            for j, a2 in enumerate(a1):
                    output[i,j] = [a3, b[i,j]]
        return output

    @staticmethod
    @jit(nopython=True)
    def zip_2d(a,b):
        '''
            We can jit this to make it a bit faster.
            Turns the uproot hist into the correct boost format by combining means/variances into one object
        '''
        output = np.zeros((*a.shape,2))
        for i, a1 in enumerate(a):
            for j, a2 in enumerate(a1):
                output[i,j] = [a3, b[i,j]]
        return output

    @staticmethod
    @jit(nopython=True)
    def zip_3d(a,b):
        '''
            We can jit this to make it a bit faster.
            Turns the uproot hist into the correct boost format by combining means/variances into one object
        '''
        output = np.zeros((*a.shape,2))
        for i, a1 in enumerate(a):
            for j, a2 in enumerate(a1):
                for k, a3 in enumerate(a2):
                    output[i,j,k] = [a3, b[i,j,k]]
        return output

    def plotprojections(self):
        import matplotlib.pyplot as plt 
        import matplotlib 

        fig,ax = plt.subplots(1,self.ndim,figsize=(15,5))
        for i, axi in enumerate(ax):
            plt.sca(axi)
            hi = self.h.project(i)
            # print(hi.axes[0].centers)
            # print(hi.view().value)
            plt.title("Axis "+str(i))
            plt.plot( hi.axes[0].centers, hi.view().value )
            plt.yscale("log")
        # plt.show()

        return (fig,ax)

    def plothist2d(self, axes=None, ax=None, SetLogz=True, linthresh=1, hist=None):
        import matplotlib.pyplot as plt 
        import matplotlib 
        if(ax is None):
            fig,ax = plt.subplots(figsize=(8,5))
        plt.sca(ax)
        if(SetLogz):
            norm = matplotlib.colors.SymLogNorm(linthresh=linthresh) #maybe replace with symlognorm?
        else:
            norm = None

        if((self.ndim == 2)):
            return plt.pcolormesh(*self.h.axes.edges.T, self.h.view().value.T, norm=norm)
        elif(axes is not None and hist is  None):
            assert len(axes) == 2
            h = self.h.project(*axes)
            return plt.pcolormesh(*h.axes.edges.T, h.view().value.T, norm=norm)
        elif(hist is not None): #add utility for ploting any hist, i.e. Kloss histogram
            h = hist
            return plt.pcolormesh(*h.axes.edges.T, h.view().T, norm=norm)
        else:
            raise ValueError("Histogram axes are incorrect. Try again with axes=[0,1] for example")


    def do_pileup_correction(self):
        print("performing pileup correction")

        self.compute_unscaled_double(self.h)
        self.compute_unscaled_triple(self.h)

    @staticmethod
    @jit(nopython=True)
    def compute_unscaled_double(h):
        return h


    def rootHistToNumpy(self, hist3d):        
        shape = (hist3d.GetNbinsX(), hist3d.GetNbinsY(), hist3d.GetNbinsZ())
        array = np.empty(shape)
        err_array = np.empty(shape)

        retcode = r.hist3dToNumpyArray(
            array, err_array, array.size, hist3d.GetName().encode())
        
        return array, err_array