import pint
from pint.facets.measurement.objects import Measurement
u = pint.UnitRegistry()
pint.set_application_registry(u)
u.setup_matplotlib(True)
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.odr as odr

def setAxis(x, y):
    global xunit, yunit
    xunit = x
    yunit = y

def doRead(filename):
    with open(f"data/{filename}.csv") as file:
        reader = csv.reader(file, delimiter=";", quotechar='"')
        names = next(reader)
        units = [u(unit) for unit in next(reader)]
        errors = [float(error) for error in next(reader)]
        values = {name: [] for name in names}
        for row in reader:
            for i, name in enumerate(names):
                values[name].append(pint.Measurement(float(row[i]), errors[i], units[i]))
    return {name: np.array(values[name]) for name in names}

def fit(model, x, y, estimate, ax = None, label="", exclude=[]):
    kvargs = {}
    xunit = [x.units for x in x[0]] if x[0] is tuple else x[0].units
    yunit = [y.units for y in y[0]] if y[0] is tuple else y[0].units
    data = odr.RealData(
        [x[i].value / xunit for i in range(len(x)) if i not in exclude],
        [y[i].value / yunit for i in range(len(y)) if i not in exclude],
        sx=[x[i].error / xunit for i in range(len(x)) if i not in exclude],
        sy=[y[i].error / yunit for i in range(len(y)) if i not in exclude],)
    betaUnits = [estimate.units for estimate in estimate]

    def myModel(beta, x):
        return model(x * xunit, *([beta[i] * betaUnits[i] for i in range(len(beta))])).to(yunit) / yunit
    runner = odr.ODR(data, odr.Model(myModel), beta0=[estimate[i] / betaUnits[i] for i in range(len(estimate))], **kvargs)
    out = runner.run()
    beta = [out.beta[i] * betaUnits[i] for i in range(len(out.beta))]
    sd_beta = [out.sd_beta[i] * betaUnits[i] for i in range(len(out.sd_beta))]
    result = [pint.Measurement(beta[i], sd_beta[i]) for i in range(len(beta))]
    if ax is not None:
        xs = np.linspace(min(x).value, max(x).value, 100)
        ys = model(xs, *beta)
        fullabel = label.format(*result)
        ax.plot(xs, ys, label=fullabel)
    return result

plt.rcParams["text.latex.preamble"] = "\\usepackage{siunitx} \\usepackage{gensymb}"

errorbarPreset = {
    "ms":10, "mfc":"none", "marker": "o", "linestyle": "none",
}

def errorbar(ax, x, y, **kwargs):
    xValue = [x.value if isinstance(x, Measurement) else x for x in x]
    xError = [x.error if isinstance(x, Measurement) else 0 for x in x]
    yValue = [y.value if isinstance(y, Measurement) else y for y in y]
    yError = [y.error if isinstance(y, Measurement) else 0 for y in y]
    kwargs = dict(errorbarPreset, **kwargs)
    ax.errorbar(xValue, yValue, xerr=xError, yerr=yError, **kwargs)

import scipy
def average(x):
    xError = 0
    if isinstance(x[0], Measurement):
        xerror = x[0].error
        x = [x[i].value for i in range(len(x))]
    unit = x[0].units
    x = np.array([x[i].to(unit) / unit for i in range(len(x))])
    xerror = xerror.to(unit) / unit
    xbar = sum(x) / len(x)
    sigma = np.sqrt(sum((x - xbar)**2) / (len(x)-1))
    return Measurement(xbar, np.sqrt(xerror**2 + (sigma * scipy.stats.t.ppf((1 + 0.682) / 2, len(x)-1) / np.sqrt(len(x)))**2)) * unit

def varianceWeightedMean(x):
    unit = x[0].units
    dx = np.array([(x.error / unit).to(u.dimensionless) for x in x])
    x = np.array([(x.value / unit).to(u.dimensionless) for x in x])
    w = 1/(dx**2)
    mean = sum(x * w) / sum(w)
    uint = np.sqrt(1 / (sum(w)))
    uext = np.sqrt(sum(w * (x-mean)**2) / ((len(x)-1) * sum(w)))
    return Measurement(mean, max(uint, uext)) * unit