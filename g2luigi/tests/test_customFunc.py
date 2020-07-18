from g2Fitter import *
from root_to_boost import *
import uproot


infile = "/home/jlab/g-2/omega_a_KS/data/results_clustersAndCoinc_unrandomized_July7_pileup_corrected.root"
f = r.TFile(infile)
h3 = f.Get("clustersAndCoincidences/corrected").Clone()
h3.GetYaxis().SetRangeUser(1700,3200)
h = h3.Project3D("x").Clone()
print(h)


input_hist = rootToBoost.TH1ToBoost(None, h)


print((input_hist.axes[0].centers))


# fig,ax = plt.subplots(figsize=(15,5))
# plt.plot(input_hist.axes[0].centers, input_hist.view().value)
# plt.yscale("log")
# plt.xlim(30,400)
# plt.show()


def custom_fit_func(x,p):
    a = p[0]
    b = p[1]

    return a + b*x

def second_custom_func(x,p):
    return p[0] + p[1]*np.exp(p[2]*x)

xlims=[30,650]
whichFit='custom'
whichCost='LeastSquares'
initialGuess=[  100,-1  ]
initialGuess2 = [1000000, 1000000, -1/64.]
blindingString='wow what a crazy blinding string this is!!!'

fitter = g2Fitter(whichFit, whichCost, blindingString, input_hist, 
                  initialGuess2, xlims, custom_func=second_custom_func)

fitter.do_fit()

fitter.plot_result()

print("All done!")


