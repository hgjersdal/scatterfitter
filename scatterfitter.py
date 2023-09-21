import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math


def read_data():
    """Get data from CSV"""
    with open('g4QBERTZ_5e8_TH2_r500_binSize1mm2Img.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

        data_array = np.array(data, dtype=float)
        return data_array[250:750, 250:750]

data = read_data()
print(data)
plt.matshow(data)
plt.show()

# The Poisson variance is the expexted mean. The expected mean is the number of hits in the bin, or 1.
var = data
var[var<1] = 1


def gaussian(rsqr, sigma, A):
    """
    2D gaussian, but we assume it transforms as 1D Gaussian with radius from center and single sigma
    So we are really fitting 2D data to a 1D Gauss, where r*r = x*x + y*y
    """
    return (A / (sigma**2 * 2 * np.pi)) * np.exp(- rsqr/(sigma*sigma))


def multi_gauss(rsqr, params):
    """ Several Gausses on top of each other """
    val = 0
    for i in range(len(params)//2):
        val += gaussian(rsqr, params[i*2], params[i*2+1])
    return val


def make_fit(params):
    """ Make a matric from a set of gaussian parameters """
    fit = np.zeros(data.shape)
    for x in range(500):
        for y in range(500):
            rsq = (x-249.5)**2 + (y-249.5)**2
            fit[x, y] += multi_gauss(rsq, params)
    return fit


def chi_squared(params):
    """
    chi squared is squared residuals divided by the variance,
    this is the function we want to optimize """
    fit = make_fit(params)
    chisq = (fit - data)**2/var
    summed = chisq.sum()
    print(str(params) + "," + str(summed))
    return summed


def make_scatter_plot():
    radii = np.zeros(500*500)
    vals = np.zeros(500*500)
    index = 0
    for x in range(500):
        for y in range(500):
            radius = math.sqrt((x-249)**2 + (y-249)**2)
            if y < 249:
                radius *= -1
            radii[index] = radius
            vals[index] = data[x, y]
            index += 1
            # print(str(radius) + ', ' + str(data[x, y]))
    return radii, vals



# Make a scatterplot of the data
radii, vals = make_scatter_plot()
plt.scatter(radii, vals, s=4)
print(radii)


# Plot the Gaussian sum desctibed by params
def mult_gauss_plot(params, xs, lbl):
    def mga(x):
        return multi_gauss(x**2, params)
    ys = np.array([mga(r) for r in xs])
    ys[ys < 1] = 1
    plt.plot(xs, ys, label=lbl)



xs = np.linspace(-250, 250, 500)

params6 = [1.03538463e+01, 8.03250571e+08, 1.50521177e+01, 1.38563360e+08, 2.48976758e+01, 2.92682121e+07, 2.83451220e+02, 2.79124680e+06, 4.50377575e+01, 7.44242623e+06, 9.18093612e+01, 2.05398347e+06]
params5 = [1.04182992e+01, 8.30840063e+08, 1.63239725e+01, 1.24554259e+08,  3.08056288e+01, 2.05431747e+07, 2.34480576e+02, 2.80489374e+06,  6.61299463e+01, 4.24356058e+06]
params4 = [1.05377247e+01, 8.67841985e+08, 1.86380093e+01, 9.94789108e+07, 4.38545434e+01, 1.20832954e+07, 1.82795365e+02, 3.05836237e+06]
params3 = [1.06933752e+01, 9.01105354e+08, 2.20956316e+01, 7.34603862e+07, 7.66777886e+01, 6.07651451e+06]
params2 = [1.08207672e+01, 9.20774754e+08, 2.56767779e+01, 5.65140248e+07]
params1= [1.12345079e+01, 9.58922768e+08]

mult_gauss_plot(params6, xs, "6")
mult_gauss_plot(params5, xs, "5")
mult_gauss_plot(params4, xs, "4")
mult_gauss_plot(params3, xs, "3")
mult_gauss_plot(params2, xs, "2")
mult_gauss_plot(params1, xs, "1")

plt.yscale("log")
plt.legend()
plt.show()


plt.plot(xs, data[249, :])
mult_gauss_plot(params6, xs, "6")
mult_gauss_plot(params5, xs, "5")
mult_gauss_plot(params4, xs, "4")
mult_gauss_plot(params3, xs, "3")
mult_gauss_plot(params2, xs, "2")
mult_gauss_plot(params1, xs, "1")
plt.yscale("log")
plt.legend()
plt.show()


fit = make_fit(params1)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

fit = make_fit(params3)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

fit = make_fit(params6)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()



params = params6

while True:
 result = minimize(chi_squared, params, method='Nelder-Mead')
 # result = minimize(chi_squared, params)
 # result = minimize(chi_squared, params, method='SLSQP')
 if result.success:
     fitted_params = result.x
     print(fitted_params)
     break
 else:
     print(result.message)
     params = result.x
