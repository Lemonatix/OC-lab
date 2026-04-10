global xunit, yunit
from common import *

def read(filename):
    globals().update((name, value) for name, value in doRead(filename).items())
setAxis(u.meter, u.second)
read("demo")

def model(t, a):
    return 1/2 * a * t**2

fig, ax = plt.subplots()
errorbar(ax, x, y)
a = fit(model, x, y, [1 * u.meter/u.second**2], ax)
ax.set(xlabel=r'$x\ [\mathrm{m}]$', ylabel=r'$y\ [\mathrm{s^{-2}}]$')
fig.savefig("files/demo.pdf")

print(a)