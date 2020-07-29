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


xlims=[30,650]
whichFit='13par'
whichCost='LeastSquares'
initialGuess=[  1.41236293e+07,  6.44282734e+01,  3.76792699e-01, -6.61887714e+01,
                5.30832191e+00, -5.64781236e-01, -4.78112413e-02,  1.92354573e-02,
                2.96476900e+00,  4.06198032e-01,  1.06624461e+00,  1.33384013e+00,
                4.89043459e-01]
blindingString='wow what a crazy blinding string this is!!!'
limits=[None, (0.1,1000), None, None, None, (-1,1) ,(-1,1), (-1,1), (0.1,1000),None,None,None,None ]

fitter = g2Fitter(whichFit, whichCost, blindingString, input_hist, 
                  initialGuess, xlims, do_iterative_fit=True, fit_list=[5,8], fit_limits=limits,
                  final_unlimited_fit=False)

fitter.do_fit()

fitter.plot_result()

print("All done!")


