import numpy as np
import matplotlib.pyplot as plt

def bspline_curve3D(precision, knot_vector, points):
    if precision % 2 == 0:
        precision = precision + 1

    compute_nr_basis_functions = lambda knot_vector, p: len(knot_vector) - p - 1
    mesh = lambda a, c: np.linspace(a, c, precision + 1)

    p = compute_p(knot_vector)
    t = check_sanity(knot_vector, p)

    if not t:
        print("Poorly constructed knot_vector")
        return

    nr = compute_nr_basis_functions(knot_vector, p)

    n, m = points.shape

    if n != nr or m != 3:
        print("Poorly constructed points array - expected N x 3")
        return

    x_begin = knot_vector[0]
    x_end = knot_vector[-1]
    x = mesh(x_begin, x_end)

    res_x = np.zeros(len(x))
    res_y = np.zeros(len(x))
    res_z = np.zeros(len(x))

    for ind in range(nr):
        spline = compute_spline(knot_vector, p, ind + 1, x)
        print()
        res_x = res_x + (spline * points[ind, 0])
        res_y = res_y + (spline * points[ind, 1])
        res_z = res_z + (spline * points[ind, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(res_x, res_y, res_z, 'r', linewidth=1.5)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=36)

    ax.set_xlim(-0.5, 2)
    ax.set_ylim(0, 3)
    ax.set_zlim(-1.5, 1.5)

    plt.grid(True)
    plt.show()

def compute_p(knot_vector):
    initial = knot_vector[0]
    kvsize = len(knot_vector)
    p = 0

    while (p + 2 <= kvsize) and (initial == knot_vector[p + 1]):
        p = p + 1
    return p

def compute_splines(knot_vector, p, nr, x):
    y = compute_spline(knot_vector, p, nr, x)
    return y

def compute_spline(knot_vector, p, nr, x):
    fC = lambda x, a, b: (b != a) * (x - a) / (b - a)
    fD = lambda x, c, d: (d != c) * (d - x) / (d - c)

    a = knot_vector[nr - 1]
    b = knot_vector[nr + p - 1]
    c = knot_vector[nr]
    d = knot_vector[nr + p]

    if p == 0:
        y = np.where(x < a, 0, np.where((x >= a) & (x <= d), 1, 0))
        return y

    lp = compute_spline(knot_vector, p - 1, nr, x)
    rp = compute_spline(knot_vector, p - 1, nr + 1, x)

    if a == b:
        y1 = np.where(x < a, 0, np.where((x >= a) & (x <= b), 1, 0))
    else:
        y1 = fC(x, a, b) * ((x >= a) & (x <= b))

    if c == d:
        y2 = np.where(x < c, 0, np.where((x >= c) & (x <= d), 1, 0))
    else:
        y2 = fD(x, c, d) * ((x >= c) & (x <= d))

    y = lp * y1 + rp * y2

    if nr == 1 and x[0] == knot_vector[0]:
        y[0] = 1
    elif nr == len(knot_vector) - p - 1 and x[-1] == knot_vector[-1]:
        y[-1] = 1

    return y

def check_sanity(knot_vector, p):
    initial = knot_vector[0]
    kvsize = len(knot_vector)
    t = True
    counter = 1

    for i in range(p + 1):
        if initial != knot_vector[i]:
            t = False
            return t

    for i in range(p + 1, kvsize - p - 1):
        if initial == knot_vector[i]:
            counter = counter + 1
            if counter > p:
                t = False
                return t
        else:
            initial = knot_vector[i]
            counter = 1

    initial = knot_vector[-1]

    for i in range(kvsize - p - 1, kvsize):
        if initial != knot_vector[i]:
            t = False
            return t

    for i in range(kvsize - 1):
        if knot_vector[i] > knot_vector[i + 1]:
            t = False
            return t

    return t

pointsK = np.array([
    [0, 3, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 1.5, 0],
    [0, 1.5, 0],
    [1.5, 3, 0],
    [1.5, 3, 0],
    [0, 1.5, 0],
    [0, 1.5, 0],
    [1.5, 0, 0]
])
knot_vectorK = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 5])

pointsG = np.array([
    [1, 1.5, 0],
    [1.5, 1.5, 0],
    [1.5, 1.5, 0],
    [1.5, 0, 0],
    [0, 0, 0],
    [0, 3, 0],
    [1.5, 2.5, 0]
])
knot_vectorG = np.array([0, 0, 0, 1, 2, 3, 4, 5, 5, 5])

bspline_curve3D(100, knot_vectorK, pointsK)
bspline_curve3D(100, knot_vectorG, pointsG)