import uproot
import boost_histogram as bh
import numba
from numba import jit
import numpy as np


class g2Histogram():
    '''
        Reads in a root histogram into the boost format using uproot. 
        Uses the file type to determine how to read this in
    '''
    def __init__(self, infile, histname):
        
        if(".root" in infile):
            self.read_from_root(infile, histname)

    def read_from_root(self, infile, histname):
        print("Reading in hist "+str(histname)+" from root file "+str(infile))
        import uproot

        #Following ROOT FILE FORMAT EXAMPLE in boost-histogram docs: 
        #   https://readthedocs.org/projects/boost-histogram/downloads/pdf/latest/

        root_file = uproot.open(infile)
        uproot_hist = root_file[histname]
        print(uproot_hist)
        print(type(uproot_hist.edges))

        self.ndim = len(uproot_hist.edges)

        boost_axes = []
        for axis in uproot_hist.edges:
            boost_axes.append( bh.axis.Variable(axis) )
        # uproot_hist.show()

        self.h = bh.Histogram(*boost_axes, storage=bh.storage.Weight() )

        print("starting zip")

        self.zipped = self.zip(uproot_hist.allvalues, uproot_hist.allvariances )
        self.h[:,:,:] = self.zipped
        print("done!")

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
                theselims.append( [self.h.axes[i].index(xlims[0]), self.h.axes[i].index(xlims[1])] )
            else:
                theselims.append( [0, self.h.axes[i].size])
        print("Projecting with:", [xlims, ylims,zlims],"->",theselims)

        #make projection
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
            print(hi.axes[0].centers)
            print(hi.view().value)
            plt.title("Axis "+str(i))
            plt.plot( hi.axes[0].centers, hi.view().value )
            plt.yscale("log")
        # plt.show()

        return (fig,ax)

    def plothist2d(self, axes=None, ax=None, SetLogz=True, linthresh=1):
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
        elif(axes is not None):
            assert len(axes) == 2
            h = self.h.project(*axes)
            return plt.pcolormesh(*h.axes.edges.T, h.view().value.T, norm=norm)
        else:
            raise ValueError("Histogram axes are incorrect. Try again with axes=[0,1] for example")


    def do_pileup_correction(self):
        print("performing pileup correction")