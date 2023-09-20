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

        # d2 = data_array
        # d2[d2>0] = 1
        # plt.matshow(d2)
        # plt.show()
        
        return data_array[250:750, 250:750]

data = read_data()
print(data)
plt.matshow(data)
plt.show()

# The Poisson variance is the expexted mean. The expected mean is the number of hits in the bin, or 1.
var = data
var[var<1] = 1
# plt.matshow(var)
# plt.show()


def gaussian(rsqr, sigma, A):
    """
    2D gaussian, but we assume it transforms as 1D Gaussian with radius from center and single sigma
    So we are really fitting 2D data to a 1D Gauss, where r*r = x*x + y*y
    """
    return (A / (sigma * 2 * np.pi)) * np.exp(- rsqr/(sigma*sigma))


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
            rsq = (x-249)**2 + (y-249)**2
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


# [1.12345079e+01 8.53551197e+07],15693218.868358668
# [1.08207672e+01 8.50932965e+07 2.56767780e+01 2.20097802e+06],6510213.8494843375
# [1.06933752e+01 8.42676274e+07 2.20956314e+01 3.32465669e+06  7.66777791e+01 7.92474078e+04],4833499.049764259
params = [1.06933752e+01, 8.42676274e+07, 2.20956314e+01, 3.32465669e+06,  7.66777791e+01, 7.92474078e+04]

# fit = make_fit(params)
# plt.matshow(fit - data)
# plt.show()


# Make a scatterplot of the data
radii, vals = make_scatter_plot()
plt.scatter(radii, vals)
print(radii)


# Plot the Gaussian sum desctibed by params
def mga(r):
    rr = r**2
    return multi_gauss(rr, params)


xs = np.linspace(-250, 250, 500)
ys = np.array([mga(r) for r in xs])
plt.plot(xs, ys, color='red')
plt.yscale("log")
plt.show()


# Optimize with Nelder-Mead
# result = minimize(chi_squared, params)
result = minimize(chi_squared, params, method='Nelder-Mead')
# result = minimize(chi_squared, params, method='SLSQP')
if result.success:
    fitted_params = result.x
    print(fitted_params)
else:
    raise ValueError(result.message)
