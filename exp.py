import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np

# print(plt.get_cmap('jet', 256)(np.linspace(0, 1, 265)))
# print(type(plt.get_cmap('jet', 256)(np.linspace(0, 1, 265))))

red = plt.get_cmap('coolwarm', 256)(256)
blue = plt.get_cmap('coolwarm', 256)(0)
cmap = np.vstack(
    (
        matplotlib.colors.LinearSegmentedColormap.from_list('a', ['green', blue], 256)(np.linspace(0, 1, 256)),
        plt.get_cmap('coolwarm', 256)(np.linspace(0, 1, 256)),
        matplotlib.colors.LinearSegmentedColormap.from_list('a', [red, 'violet'], 256)(np.linspace(0, 1, 256)),
    )
)
cmap = matplotlib.colors.ListedColormap(cmap)
vmin=-80951.85105559323
vmax=80951.85105559323
vmin_original=-19453.6
vmax_original=19453.6
a = vmin
b = vmin_original
c = vmax_original
d = vmax


def foward(x, a, b, c, d):
    return np.piecewise(x, [(a <= x) & (x < b), (b <= x) & (x < c), (c <= x) & (x <= d), x < a, x > d],
                        [lambda x: 1 / 3 * (x - a) / (b - a), lambda x: 1 / 3 + 1 / 3 * (x - b) / (c - b),
                         lambda x: 2 / 3 + 1 / 3 * (x - c) / (d - c), 0, 1])


def inverse(x, a, b, c, d):
    return np.piecewise(x, [(0 <= x) & (x < 1 / 3), (1 / 3 <= x) & (x < 2 / 3), (2 / 3 <= x) & (x <= 1), x < 0, x > 1],
                        [lambda x: a + 3 * x * (b - a), lambda x: b + 3 * (x - 1 / 3) * (c - b),
                         lambda x: c + 3 * (x - 2 / 3) * (d - c), a, d])


# plt.plot(x, ff(x, a, b, c, d))
# plt.show()

norm = matplotlib.colors.FuncNorm((lambda x: foward(x, a, b, c, d), lambda x: inverse(x, a, b, c, d)), vmin=min(a, -d),
                                  vmax=max(d, -a))
print(f'{norm(0)=}')
x = np.arange(-40, 12., .1)
plt.scatter(x, x ** 3, c=x ** 3, cmap=cmap, norm=norm)
ticks = np.linspace(a, d, 15, endpoint=True)
ticks=[a,(a+b)/2,b,b/2,0,c/2,c,(c+d)/2,d]
plt.colorbar(ticks=ticks)
print(f'{red=}')
print(f'{blue=}')
plt.show()
