import law 
import json
import luigi
import os
import ROOT as r
import pickle
import boost_histogram as bh 
import ast


class rootToBoost(law.Task):
    '''
        Takes as input a root histogram with an arbetrary number of axes, makes cuts on those axes, and 
        projects the data within the cuts down into a 1D histogram. This is then stored in a boost format.
    '''

    infile = luigi.Parameter()
    histName = luigi.Parameter(default="corrected")
    campaignName = luigi.Parameter(default="test")
    outputDir = luigi.Parameter(default="./data/")

    timeAxis = luigi.Parameter(default='x')
    axisCuts = luigi.Parameter(default=None, description="['axis string', lower bound, upper bound] -> ['y', 1700, 3200]")

    def output(self):
        return law.LocalFileTarget(f"{self.outputDir}{self.campaignName}_{self.histName.replace('/','_')}.pickle")

    def run(self):
        output = self.output()
        output.parent.touch()  # creates the data/ dir

        # f = uproot.open(self.infile)
        # hi = f[self.histName]

        # print(hi)

        f = r.TFile(self.infile)
        # f.ls()
        try:
            hi = f.Get(self.histName).Clone(self.histName)
            # print(hi)
        except:
            raise ValueError("ERROR: no such histogram --- ", self.histName)

        # hi_boost = hi.to_boost()
        # print(hi_boost)

        print(ast.literal_eval(str(self.axisCuts)))
        #Make cuts on the axes and create a 1D histogram
        if(self.axisCuts is not None):
            try:
                ding = ast.literal_eval(self.axisCuts)
                print("ding", ding, ding[0])
                for axis, lowerBound, upperBound in ding:
                    print(axis,lowerBound,upperBound)
                    if('x' in axis):
                        hi.GetXaxis().SetRangeUser(lowerBound,upperBound)
                    if('y' in axis):
                        hi.GetYaxis().SetRangeUser(lowerBound,upperBound)
                    if('z' in axis):
                        hi.GetZaxis().SetRangeUser(lowerBound,upperBound)
            except:
                print("what the literal fuck")
        hi_1D = hi.Project3D(self.timeAxis).Clone(self.histName.replace("/","_")+"_1Dprojection")
        # print(hi_1D)

        #convert TH1X to boost_histogram format via aghast (https://github.com/scikit-hep/aghast)
        hi_1D_boost = self.TH1ToBoost(hi_1D)
        # print(hi_1D_boost)

        # print("View:", hi_1D_boost.view())

        data = {self.histName:hi_1D_boost}
        self.output().dump(data, formatter="pickle")

    def TH1ToBoost(self, h):
        contents = []
        errs = []
        edges = []
        for i in range(h.GetNbinsX()+2):
            contents.append(h.GetBinContent(i))
            errs.append(h.GetBinError(i))
            edges.append(h.GetBinLowEdge(i))
        edges.append(h.GetBinLowEdge(i) + h.GetBinWidth(i)) #overflow bin

        # print(contents)
        # print(errs)
        # print(edges)

        axis = bh.axis.Variable(edges=edges[1:-1])
        h2 = bh.Histogram(axis, storage=bh.storage.Weight() )
        h2[:] = [x for x in zip(contents, errs)]
        # print(h2.view())
        # print(h2.view().value)
        # h2.weights = errs

        return h2
