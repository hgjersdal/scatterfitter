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

# The Poisson variance is the expexted mean. The expected mean is the number of hits in the bin, or 1.
var = data
var[var < 1] = 1


def gaussian(rsqr, sigma, A):
    """
    2D gaussian, but we assume it transforms as 1D Gaussian with radius from center and single sigma
    So we are really fitting 2D data to a 1D Gauss, where r*r = x*x + y*y
    """
    return (A / (sigma**2 * 2 * np.pi)) * np.exp(- 0.5 * rsqr/(sigma*sigma))


def multi_gauss(rsqr, params):
    """ Several Gausses on top of each other at poit rsqr"""
    val = 0
    for i in range(len(params)//2):
        val += gaussian(rsqr, params[i*2], params[i*2+1])
    return val


def multi_gauss_v(rsqrs, params):
    """ An image of several Gausses on top of each other """
    fit = np.zeros(data.shape)
    for i in range(len(params)//2):
        fit = fit + gaussian(rsqrs, params[i*2], params[i*2+1])
    return fit


def make_fit(params):
    """ Make a matric from a set of gaussian parameters """
    x, y = data.shape
    X, Y = np.ix_(np.arange(x), np.arange(y))
    rsqs = (X-249.5)**2 + (Y-249.5)**2
    fit = multi_gauss_v(rsqs, params)
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
    return radii, vals


radii, vals = make_scatter_plot()
plt.scatter(radii, vals, s=4)


# Plot the Gaussian sum desctibed by params
def mult_gauss_plot(params, xs, lbl):
    def mga(x):
        return multi_gauss(x**2, params)
    ys = np.array([mga(r) for r in xs])
    ys[ys < 1] = 1
    plt.plot(xs, ys, label=lbl)


xs = np.linspace(-250, 250, 500)

params7 = [7.31857717e+00, 3.97705056e+08, 1.01497237e+01, 7.01126668e+07,
           1.52928704e+01, 1.84648017e+07, 2.46103530e+01, 5.32810172e+06,
           2.26335109e+02, 1.41095479e+06, 4.11094881e+01, 1.80758619e+06,
           7.98294254e+01, 7.19943862e+05]

params6 = [7.34125207e+00, 4.06583961e+08, 1.06865471e+01, 6.81982812e+07,
           1.76552528e+01, 1.45149557e+07, 3.19051335e+01, 3.70229876e+06,
           2.00592229e+02, 1.39562804e+06, 6.50184237e+01, 1.02399722e+06]

params5 = [7.38583638e+00, 4.20137154e+08, 1.15930698e+01, 6.14089183e+07,
           2.18666216e+01, 1.01749422e+07, 4.69140562e+01, 2.10508454e+06,
           1.66107609e+02, 1.40204767e+06]

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
                   266179.14829355833,  # 6
                   265711.9130558083  # 7
                   ]


mult_gauss_plot(params7, xs, "7")
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
mult_gauss_plot(params7, xs, "7")
mult_gauss_plot(params6, xs, "6")
mult_gauss_plot(params5, xs, "5")
mult_gauss_plot(params4, xs, "4")
mult_gauss_plot(params3, xs, "3")
mult_gauss_plot(params2, xs, "2")
mult_gauss_plot(params1, xs, "1")
plt.yscale("log")
plt.legend()
plt.show()


# Plot pull-values.
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

fit = make_fit(params7)
pulls = (fit - data)/np.sqrt(var)
plt.matshow(pulls)
plt.show()

params = params7


while True:
    result = minimize(chi_squared, params, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        print(fitted_params)
        break
    else:
        print(result.message)
        params = result.x
