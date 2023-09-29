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
var = np.array(data, copy=True)
var[var < 1] = 1

print("Non-zero elements " + str(np.count_nonzero(data==0)))

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


def pearson_squared(params):
    """
    Chi squared is squared residuals divided by the variance,
    this is the function we want to optimize """
    fit = make_fit(params)
    chisq = (fit - data)**2/fit
    summed = chisq.sum()
    # print(str(params) + "," + str(summed))
    return summed


def chi_squared(params):
    """
    Neyman chi squared is squared residuals divided by the variance,
    this is the function we want to optimize """
    fit = make_fit(params)
    chisq = (fit - data)**2/var
    summed = chisq.sum()
    # print(str(params) + "," + str(summed))
    return summed


def neyman_masked(params):
    fit = make_fit(params)
    chisq = (fit - data)**2/var
    mask = data == 0
    np.putmask(chisq, mask, 0)
    summed = chisq.sum()
    # print(str(params) + "," + str(summed))
    return summed



def cousin_baker(params):
    fit = make_fit(params)
    residuals = (fit - data)
    log_term = np.array(data, copy=True)
    mask = log_term > 0
    np.putmask(log_term, mask, data * np.log(data/fit))
    return 2 * (residuals + log_term).sum()


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

params8 = [7.31508754e+00, 3.96116109e+08, 1.00493318e+01, 6.98623770e+07,
           1.47987370e+01, 1.92471168e+07, 2.29644862e+01, 5.78042588e+06,
           2.42946189e+02, 1.11679717e+06, 3.66856687e+01, 2.12917705e+06,
           6.35676502e+01, 7.53493512e+05, 1.23161876e+02, 4.87599014e+05]

params7 = [7.31705034e+00, 3.97020780e+08, 1.01074326e+01, 7.00669338e+07,
           1.50981088e+01, 1.88218479e+07, 2.40142485e+01, 5.51049300e+06,
           1.98124090e+02, 1.29496185e+06, 3.96041545e+01, 1.92052448e+06,
           7.44279775e+01, 7.24189356e+05]

params6 = [7.33918945e+00, 4.05851473e+08, 1.06416420e+01, 6.84668735e+07,
           1.74580049e+01, 1.47989587e+07, 3.12816025e+01, 3.80887343e+06,
           1.83733800e+02, 1.31980161e+06, 6.23192629e+01, 1.05018218e+06]

params5 = [7.38343257e+00, 4.19494933e+08, 1.15457833e+01, 6.17891676e+07,
           2.16330745e+01, 1.03488754e+07, 4.59252187e+01, 2.16410212e+06,
           1.58109062e+02, 1.36779246e+06]

params4 = [7.46663310e+00, 4.37951696e+08, 1.31961968e+01, 4.94125531e+07,
           3.08874714e+01, 6.05731738e+06, 1.26154159e+02, 1.51871697e+06]


params3 = [7.57755562e+00, 4.54785988e+08, 1.57094875e+01, 3.62790366e+07,
           5.48427379e+01, 2.99608445e+06]

params2 = [7.66518161e+00, 4.64364098e+08, 1.82200081e+01, 2.80289789e+07]


params1 = [7.95378141e+00, 4.83261651e+08]

converged_chi2s = [11888110.087187285,  # 1
                   2756683.7670124886,  # 2
                   1088833.7137625597,  # 3
                   358515.44133009616,  # 4
                   288500.0121852888,  # 5
                   282543.28621459496,  # 6
                   282166.0194804966,  # 7
                   282156.6889343039  # 8
                   ]


if True:
    mult_gauss_plot(params8, xs, "8")
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
    mult_gauss_plot(params8, xs, "8")
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
    plt.title("N=1")
    plt.show()

    fit = make_fit(params2)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=2")
    plt.show()

    fit = make_fit(params3)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=3")
    plt.show()

    fit = make_fit(params4)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=4")
    plt.show()

    fit = make_fit(params5)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=5")
    plt.show()

    fit = make_fit(params6)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=6")
    plt.show()

    fit = make_fit(params7)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=7")
    plt.show()

    fit = make_fit(params8)
    pulls = (fit - data)/np.sqrt(var)
    plt.matshow(pulls)
    plt.title("N=8")
    plt.show()

    bins = np.arange(-5, 5.001, 0.1)
    counts, bins = np.histogram(pulls.ravel(), bins=bins)
    print(bins)
    print(counts.sum())
    plt.stairs(counts, bins)
    mult_gauss_plot([1.0, 500*500/math.sqrt(2*math.pi)], bins, "asdf")
    plt.show()


params = params6

# print(params)
while True:
    result = minimize(neyman_masked, params, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        print(fitted_params)
        print(result.fun)
        break
    else:
        print(result.message)
        params = result.x
        print(result.fun)
        print(params)
        
