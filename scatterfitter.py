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


# [1.12345079e+01 9.58922768e+08],15693218.86835867
# [1.08207483e+01 9.20767764e+08 2.56762130e+01 5.65186091e+07],6510213.87314924
# [1.07181042e+01 9.05804308e+08 2.26886486e+01 6.88118813e+07  8.24731128e+01 5.61262252e+06],4837986.627212546
# [1.05377247e+01 8.23557276e+07 1.86380095e+01 5.33742117e+06 4.38545477e+01 2.75531134e+05 1.82795339e+02 1.67310713e+04],4081402.2629849515
# [1.05357713e+01 8.23537029e+07 1.85982212e+01 5.33818541e+06 4.05107626e+01 2.75242403e+05 8.39473584e+01 2.81355032e+04 2.65996038e+02 1.04448512e+04],4035015.2437909436
# [1.05362562e+01, 8.23554263e+07, 1.85514032e+01, 5.33488480e+06, 4.02583140e+01, 2.89389124e+05, 8.66094458e+01, 2.62153542e+04, 2.68955978e+02, 1.03075160e+04],4031452.2802090473
params = [ 1.05261568e+01, 8.72872199e+08, 1.92035164e+01, 9.56196076e+07, 4.77387466e+01, 9.58393595e+06, 2.04397747e+02, 3.45646467e+06, 3.44612096e+03, 8.48815010e+04]

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
