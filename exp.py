import matplotlib.colors
from matplotlib import pyplot as plt
import numpy as np

# print(plt.get_cmap('jet', 256)(np.linspace(0, 1, 265)))
# print(type(plt.get_cmap('jet', 256)(np.linspace(0, 1, 265))))
cmap = np.vstack((plt.get_cmap('winter', 256)(np.linspace(1, 0, 256)),
                  plt.get_cmap('coolwarm', 256)(np.linspace(0, 1, 256)),
                  plt.get_cmap('autumn', 256)(np.linspace(0, 1, 256))))
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


def _forward(x, a, b, c, d):
    print('_forward')
    print(x)
    if not a <= b <= c <= d:
        raise ValueError(f'The arguments a,b,c,d have to be a<=b<=c<=d. Got {a=}, {b=}, {c=}, {d=}')
    if a <= x <= b:
        return 1 / 3 * (x - a) / (b - a)
    if b <= x <= c:
        return 1 / 3 + 1 / 3 * (x - b) / (c - b)
    if c <= x <= d:
        return 2 / 3 + 1 / 3 * (x - c) / (d - c)
    if x < a:
        return 0
    if x > d:
        return 1


forward = np.vectorize(_forward)


def _inverse(x, a, b, c, d):
    print('_inverse')
    print(x)
    if not a <= b <= c <= d:
        raise ValueError(f'The arguments a,b,c,d have to be a<=b<=c<=d. Got {a=}, {b=}, {c=}, {d=}')
    if 0 <= x <= 1 / 3:
        return a + 3 * x * (b - a)
    if 1 / 3 <= x <= 2 / 3:
        return b + 3 * (x - 1 / 3) * (c - b)
    if 2 / 3 <= x <= 1:
        return c + 3 * (x - 2 / 3) * (d - c)
    if x < 0:
        return a
    if x > 1:
        return d


inverse = np.vectorize(_inverse)

a = -40000
b = -17000
c = 17000
d = 20000
x = np.arange(-40, 30, .1)


def foward(x, a, b, c, d):
    return np.piecewise(x, [(a <= x) & (x < b), (b <= x) & (x < c), (c <= x) & (x <= d), x < a, x > d],
                        [lambda x: 1 / 3 * (x - a) / (b - a), lambda x: 1 / 3 + 1 / 3 * (x - b) / (c - b),
                         lambda x: 2 / 3 + 1 / 3 * (x - c) / (d - c), 0, 1])


def inverse(x, a, b, c, d):
    return np.piecewise(x, [(0 <= x) & (x < 1 / 3), (1 / 3 <= x) & (x < 2 / 3), (2 / 3 <= x) & (x <= 1), x < 0, x > 1],
                        [lambda x: a + 3 * x * (b - a), lambda x: b + 3 * (x - 1 / 3) * (c - b),
                         lambda x: c + 3 * (x - 2 / 3) * (d - c), a, d])
xx=.5
print(b + 3 * (xx - 1 / 3) * (c - b))

# plt.plot(x, ff(x, a, b, c, d))
# plt.show()
z = np.arange(-.2, 1.2, .01)
print(z)
print(inverse(z, a, b, c, d))
plt.plot(z, inverse(z, a, b, c, d))
plt.show()
norm = matplotlib.colors.FuncNorm((lambda x: foward(x, a, b, c, d), lambda x: inverse(x, a, b, c, d)), vmin=a, vmax=d)
plt.scatter(x, x**3, c=x**3, cmap=cmap, norm=norm)
plt.colorbar()
print(f'{red=}')
print(f'{blue=}')
plt.show()
