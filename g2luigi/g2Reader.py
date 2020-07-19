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
        self.uproot_hist = root_file[histname]
        print(self.uproot_hist)
        print(type(self.uproot_hist.edges))

        self.ndim = len(self.uproot_hist.edges)

        boost_axes = []
        for axis in self.uproot_hist.edges:
            boost_axes.append( bh.axis.Variable(axis) )
        # uproot_hist.show()

        self.h = bh.Histogram(*boost_axes, storage=bh.storage.Weight() )

        print("starting zip")

        self.zipped = self.zip_3d(self.uproot_hist.allvalues, self.uproot_hist.allvariances )
        self.h[:,:,:] = self.zipped
        print("done!")


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