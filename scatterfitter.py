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
    return (A / (sigma**2 * 2 * np.pi)) * np.exp(- 0.5 * rsqr/(sigma*sigma))


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
            radius = math.sqrt((x-249.5)**2 + (y-249.5)**2)
            if y < 249.5:
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

params7 = [1.03820983e+01, 8.13167922e+08, 1.51130599e+01, 1.36396559e+08,
           2.49682972e+01, 2.90299120e+07, 2.83680246e+02, 2.79125608e+06,
           4.51206710e+01, 7.40459841e+06, 9.19499361e+01, 2.04799451e+06,
           2.5e+02, 1.0e+04]


params6 = [1.03820983e+01, 8.13167922e+08, 1.51130599e+01, 1.36396559e+08,
           2.49682972e+01, 2.90299120e+07, 2.83680246e+02, 2.79125608e+06,
           4.51206710e+01, 7.40459841e+06, 9.19499361e+01, 2.04799451e+06]  # 266179.14829355833

params5 = [7.49590054e+00,  4.42254899e+08,  1.36589111e+01,  4.58340938e+07,
           3.32713718e+01,  5.43017201e+06,  1.55192561e+02,  1.79415604e+06,
           1.78379491e+02, 2.89999070e+05]

params4 = [7.46823288e+00, 4.38245959e+08, 1.32290890e+01, 4.91958670e+07,
           3.11006356e+01, 5.99772212e+06, 1.29497860e+02, 1.52766408e+06]


params3 = [7.57809300e+00, 4.54852976e+08, 1.57231413e+01, 3.62236540e+07,
           5.50432580e+01, 2.98460432e+06]

params2 = [7.66518162e+00, 4.64364098e+08, 1.82200081e+01, 2.80289786e+07]

params1 = [7.95378141e+00, 4.83261651e+08]

converged_chi2s = [11892952.087187286,  # 1
                   2761525.7670124862,  # 2
                   1093369.2133564858,  # 3
                   346589.4656824806,  # 4
                   272925.0193639359,  # 5
                   266179.14829355833  # 6
                   ]

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

fit = make_fit(params2)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

fit = make_fit(params3)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

fit = make_fit(params4)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

fit = make_fit(params5)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

fit = make_fit(params6)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()



params = params5


print(data.sum())

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
