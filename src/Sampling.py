"""!
@file src/Sampling.py
@package Gnowee

@defgroup Sampling Sampling

@brief Different methods to perform phase space sampling and random walks.

Design of experiment and phase space sampling methods.  Includes some
vizualization tools.

Dependencies on pyDOE.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""

import math
import random
import bisect

import matplotlib.pyplot as plt
import numpy as np

from scipy.special import gamma
from pyDOE import lhs
from numpy.random import rand, randn, choice
from GnoweeUtilities import Switch

#------------------------------------------------------------------------------#
def initial_samples(lb, ub, method, numSamp):
    """!
    @ingroup Sampling

    Generate a set of samples in a given phase space. The current methods
    available are 'random', 'nolh', 'nolh-rp', 'nolh-cdr', 'lhc', or
    'rand-wor'.

    @param lb: \e array \n
        The lower bounds of the design variable(s). \n
    @param ub: \e array \n
        The upper bounds of the design variable(s). \n
    @param  method: \e string \n
        String representing the chosen sampling method. Valid options are:
        'random', 'nolh', 'nolh-rp', 'nolh-cdr', 'lhc', 'random-wor'. \n
    @param numSamp: \e integer \n
        The number of samples to be generated.  Ignored for nolh algorithms. \n

    @return \e array: The list of coordinates for the sampled phase space. \n
    """

    assert len(lb) == len(ub), ('Lower and upper bounds have different #s '
                           'of design variables in initial_samples function.')
    assert method == 'random' or method == 'nolh' or method == 'nolh-rp' \
           or method == 'nolh-cdr' or method == 'lhc' or method == 'rand-wor',\
                    'An invalid method was specified for the initial_samples.'
    if method == 'nolh' or method == 'nolh-rp' or method == 'nolh-cdr':
        assert len(ub) >= 2 and len(ub) <= 29, ('The Phase space dimensions '
                             'are outside of the bounds for initial_samples.')

    for case in Switch(method):
        if case('random'):
            s = np.zeros((numSamp, len(lb)))
            for i in range(0, numSamp, 1):
                s[i, :] = (lb+(ub-lb)*rand(len(lb)))
            break

        # Random without replacement - designed for combinatorial variables
        if case('rand-wor'):
            s = np.zeros((numSamp, len(lb)))
            for i in range(0, numSamp, 1):
                s[i, :] = choice(len(ub), size=len(ub), replace=False)
            break

        # Standard nearly-orthoganal latin hypercube call
        if case('nolh'):
            dim = len(ub)
            m, q, r = params(dim)
            conf = range(q)
            if r != 0:
                remove = range(dim - r, dim)
                nolh = NOLH(conf, remove)
            else:
                nolh = NOLH(conf)
            s = np.array([list(lb+(ub-lb)*nolh[i, :]) for i in \
                                                       range(len(nolh[:, 0]))])
            break

        # Nearly-orthoganal latin hypercube call with random permutation for
        # removed colummns
        if case('nolh-rp'):
            dim = len(ub)
            m, q, r = params(dim)
            conf = random.sample(range(q), q)
            if r != 0:
                remove = random.sample(range(q-1), r)
                nolh = NOLH(conf, remove)
            else:
                nolh = NOLH(conf)
            s = np.array([list(lb+(ub-lb)*nolh[i, :]) for i in \
                                                       range(len(nolh[:, 0]))])
            break

        # Nearly-orthoganal latin hypercube call with Cioppa and De Rainville
        # permutations
        if case('nolh-cdr'):
            dim = len(ub)
            m, q, r = params(dim)
            (conf, remove) = get_cdr_permutations(len(ub))
            if remove != []:
                nolh = NOLH(conf, remove)
            else:
                nolh = NOLH(conf)
            s = np.array([list(lb+(ub-lb)*nolh[i, :]) for i in \
                                                       range(len(nolh[:, 0]))])
            break

        # Latin hypercube sampling
        if case('lhc'):
            # Valid criterion are 'corr','center','maximum','centermaximum'
            tmp = lhs(len(lb), samples=numSamp, criterion="center")
            s = np.array([list(lb+(ub-lb)*tmp[i, :]) for i in \
                                                       range(len(tmp[:, 0]))])
            break

        if case():
            print "Somehow you evaded my assert statement - good job!", \
                  " However, you still need to use a valid method string."

    return s

#------------------------------------------------------------------------------#
def plot_samples(s):
    """!
    @ingroup Sampling

    Plot the first 2 and 3 dimensions on the sample distribution. Can't plot
    the full hyperspace yet.  Produces a very simple plot for visualizing the
    difference in the sampling methods.

    @param s: \e array \n
        The list of coordinates for the sampled phase space. \n
    """

    assert len(s[0, :]) >= 2, ('The Phase space dimensions are less than two.',
                               ' Need at least two to plot.')

    fig = plt.figure(1)
    if len(s[0, :]) >= 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(s[:, 0], s[:, 1], s[:, 2])
    fig = plt.figure(2)
    plt.scatter(s[:, 0], s[:, 1])
    plt.show()
    return

#------------------------------------------------------------------------------#
def levy(nc, nr=0, alpha=1.5, gam=1, n=1):
    """!
    @ingroup Sampling

    Sample the Levy distribution given by

    \f$ L_{\alpha,\gamma}(z)=\frac{1}{\pi}\int \limits_{0}^{+\infty}
        e^{-\gamma q^{\alpha}} \cos(qz) dq \f$

    using the Mantegna algoritm outlined in "Fast, Accurate Algorithm for
    Numerical Simulation of Levy Stable Stochastic Processes."

    @param nc: \e integer \n
        The number of columns of Levy values for the return array.
    @param nr \e integer \n
        The number of rows of Levy values for the return array. \n
    @param alpha \e float \n
        Levy exponent - defines the index of the distribution and controls
        scale properties of the stochastic process. \n
    @param gam: \e float \n
        Gamma - Scale unit of process for Levy flights. \n
    @param n: \e integer \n
        Number of independent variables - can be used to reduce Levy
        flight sampling variance. \n

    @return \e array:  Array representing the levy flights for each nest.
    """

    assert alpha > 0.3 and alpha < 1.99, 'Valid range for alpha is [0.3:1.99].'
    assert gam >= 0, 'Gamma must be positive'
    assert n >= 1, 'n Must be positive'

    # Calculate Levy distribution via Mantegna Algorithm
    invalpha = 1./alpha
    sigx = ((gamma(1.+alpha)*np.sin(np.pi*alpha/2.)) /(gamma((1.+alpha)/2) \
              *alpha*2.**((alpha-1.)/2.)))**invalpha
    if nr != 0:
        v = sigx*randn(n, nr, nc) \
           /(abs(randn(n, nr, nc))**invalpha)
    else:
        v = sigx*randn(n, nc) \
        /(abs(randn(n, nc))**invalpha)
    kappa = (alpha*gamma((alpha+1.)/(2.*alpha)))/gamma(invalpha) \
            *((alpha*gamma((alpha+1.)/2.))/(gamma(1.+alpha) \
              *np.sin(np.pi*alpha/2.)))**invalpha
    p = [-17.7767, 113.3855, -281.5879, 337.5439, -193.5494, 44.8754]
    c = np.polyval(p, alpha)
    w = ((kappa-1.)*np.exp(-abs(v)/c)+1.)*v

    if n > 1:
        z = (1/n**invalpha)*sum(w)
    else:
        z = w

    z = gam**invalpha*z
    if nr != 0:
        z = z.reshape(nr, nc)
    else:
        z = z.reshape(nc)

    return z

#------------------------------------------------------------------------------#
def tlf(numRow=1, numCol=1, alpha=1.5, gam=1., cutPoint=10.):
    """!
    @ingroup Sampling

    Samples from a truncated Levy flight distribution (TLF) according to
    Manegna, "Stochastic Process  with Ultraslow Convergence to a Gaussian:
    The Truncated Levy Flight" to map a levy distribution onto the interval
    [0,1].

    @param numRow: \e integer \n
        Number of rows of Levy flights to sample. \n
    @param numCol: \e integer \n
        Number of columns of Levy flights to sample. \n
    @param alpha: \e float \n
        Levy exponent - defines the index of the distribution and controls
        scale properties of the stochastic process. \n
    @param gam: \e float \n
        Gamma - Scale unit of process for Levy flights. \n
    @param cutPoint: \e float \n
        Point at which to cut sampled Levy values and resample. \n

    @return \e array: Array representing the levy flights on the interval
        (0,1). \n
    """

    # Draw numRow x numCol samples from the Levy distribution
    z = abs(levy(numRow, numCol)/cutPoint).reshape(numRow, numCol)

    # Resample values above the range (0,1)
    for i in range(len(z)):
        for j in range(len(z[i])):
            while z[i, j] > 1:
                z[i, j] = abs(levy(1, 1, alpha=alpha, gam=gam) \
                              /cutPoint).reshape(1)

    return z

#------------------------------------------------------------------------------#
def NOLH(conf, remove=None):
    """!
    @ingroup Sampling

    This library allows to generate Nearly Orthogonal Latin Hypercubes (NOLH)
    according to Cioppa (2007) and De Rainville et al. (2012) and reference
    therein.

    https://pypi.python.org/pypi/pynolh

    Constructs a Nearly Orthogonal Latin Hypercube (NOLH) of order *m* from
    a configuration vector *conf*. The configuration vector may contain either
    the numbers in $ [0 q-1] $ or $ [1 q] $ where $ q = 2^{m-1} $.
    The columns to be *removed* are also in $ [0 d-1] $ or $ [1 d] $
    where

    $ d = m + \binom{m-1}{2} $

    is the NOLH dimensionality.

    The whole library is incorporated here with minimal modification for
    commonality and consolidation of methods.

    @param conf: \e array \n
        Configuration vector. \n
    @param remove: \e array \n
        Array containing the indexes of the colummns to be removed from conf
        vector. \n

    @return \e array: Array containing nearly orthogonal latin hypercube
         sampling. \n
    """

    I = np.identity(2, dtype=int)
    R = np.array(((0, 1), (1, 0)), dtype=int)

    if 0 in conf:
        conf = np.array(conf) + 1

        if remove is not None:
            remove = np.array(remove) + 1

    q = len(conf)
    m = math.log(q, 2) + 1
    s = int(m + (math.factorial(m - 1) / (2 * math.factorial(m - 3))))

    # Factorial checks if m is an integer
    m = int(m)

    A = np.zeros((q, q, m - 1), dtype=int)
    for i in range(1, m):
        Ai = 1
        for j in range(1, m):
            if j < m - i:
                Ai = np.kron(Ai, I)
            else:
                Ai = np.kron(Ai, R)

        A[:, :, i-1] = Ai

    M = np.zeros((q, s), dtype=int)
    M[:, 0] = conf

    col = 1
    for i in range(0, m - 1):
        for j in range(i + 1, m):
            if i == 0:
                M[:, col] = np.dot(A[:, :, j-1], conf)
            else:
                M[:, col] = np.dot(A[:, :, i-1], np.dot(A[:, :, j-1], conf))
            col += 1

    S = np.ones((q, s), dtype=int)
    v = 1
    for i in range(1, m):
        for j in range(0, q):
            if j % 2**(i-1) == 0:
                v *= -1
            S[j, i] = v

    col = m
    for i in range(1, m - 1):
        for j in range(i + 1, m):
            S[:, col] = S[:, i] * S[:, j]
            col += 1

    T = M * S

    keep = np.ones(s, dtype=bool)
    if remove is not None:
        keep[np.array(remove) - 1] = [False] * len(remove)

    return (np.concatenate((T, np.zeros((1, s)), -T), axis=0)[:, keep] + q) \
            / (2.0 * q)

#------------------------------------------------------------------------------#
def params(dim):
    """!
    @ingroup Sampling

    Returns the NOLH order $m$, the required configuration length $q$
    and the number of columns to remove to obtain the desired dimensionality.

    @param dim: \e integer \n
        The dimension of the space. \n
    """
    m = 3
    #Original version has three here, but this failed each time the # of
    # samples required switched (ie at dim=3,7,11,etc)
    s = 1
    q = 2**(m-1)

    while s < dim:
        m += 1
        s = m + math.factorial(m - 1) / (2 * math.factorial(m - 3))
        q = 2**(m-1)

    return m, q, s - dim

#------------------------------------------------------------------------------#
def get_cdr_permutations(dim):
    """!
    @ingroup Sampling

    Generate a set of CDR permulations for NOLH.

    @param dim: \e integer \n
        The dimension of the space. \n

    @return \e array: A configuration vector. \n
    @return \e array: Array containing the indexes of the colummns to be
        removed from conf vector. \n
    """

    assert dim >= 2 and dim <= 29, ('The Phase space dimensions are outside ',
                                'of the bounds for CDR Permutations.')

    # Permutation and columns to remove given by Cioppa
    C_CONF = {
        2 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 3, 4, 6, 7]),
        3 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 2, 3, 6]),
        4 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 3, 6]),
        5 : ([1, 2, 8, 4, 5, 6, 7, 3], [1, 6]),
        6 : ([1, 2, 8, 4, 5, 6, 7, 3], [1]),
        7 : ([1, 2, 8, 4, 5, 6, 7, 3], [])
    }

    # Permutation and columns to remove given by De Rainville et al.
    EA_CONF = {
        8  : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10],
              [1, 3, 10]),
        9  : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10],
              [6, 10]),
        10 : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10],
              [10]),
        11 : ([4, 14, 1, 2, 16, 13, 5, 8, 12, 9, 6, 7, 11, 3, 15, 10],
              []),
        12 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8,
               24, 29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4],
              [2, 4, 5, 11]),
        13 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8,
               24, 29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4],
              [3, 6, 14]),
        14 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], [4, 5]),
        15 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], [6]),
        16 : ([5, 13, 19, 23, 28, 10, 12, 32, 17, 2, 30, 15, 6, 31, 21, 8, 24,
               29, 9, 14, 11, 22, 18, 25, 3, 1, 20, 7, 27, 16, 26, 4], []),

        17 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60],
              [8, 11, 12, 14, 17]),
        18 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60],
              [8, 11, 12, 17]),
        19 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60],
              [10, 15, 22]),
        20 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60],
              [8, 12]),
        21 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], [15]),
        22 : ([7, 8, 51, 3, 40, 44, 29, 19, 61, 43, 26, 48, 20, 52, 4, 49, 2,
               57, 31, 30, 24, 23, 56, 50, 18, 59, 63, 37, 38, 21, 54, 9, 46,
               27, 36, 1, 10, 42, 13, 55, 15, 25, 22, 45, 41, 39, 53, 34, 6, 5,
               2, 58, 16, 28, 64, 14, 47, 33, 12, 35, 62, 17, 11, 60], []),

        23 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [18, 20, 21, 24, 27, 29]),
        24 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [4, 15, 18, 24, 27]),
        25 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [21, 26, 27, 29]),
        26 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [26, 27, 29]),
        27 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [27, 29]),
        28 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [20]),
        29 : ([9, 108, 39, 107, 62, 86, 110, 119, 46, 43, 103, 71, 123, 91, 10,
               13, 126, 63, 83, 47, 100, 54, 23, 16, 124, 45, 27, 4, 93, 74, 76,
               90, 30, 81, 77, 53, 116, 49, 104, 6, 70, 82, 26, 118, 55, 79, 32,
               109, 57, 31, 22, 101, 44, 87, 121, 7, 37, 56, 89, 115, 25, 92,
               85, 20, 58, 52, 3, 11, 106, 17, 117, 38, 78, 28, 59, 96, 18, 97,
               50, 114, 112, 60, 84, 1, 12, 61, 98, 128, 14, 42, 64, 105, 68,
               75, 111, 34, 141, 65, 99, 2, 19, 33, 35, 94, 51, 122, 127, 36,
               125, 80, 73, 8, 24, 21, 88, 48, 69, 66, 40, 15, 29, 113, 72, 5,
               95, 120, 6, 102], [])
    }

    # Create dictionary
    CONF = dict()
    CONF.update(C_CONF)
    CONF.update(EA_CONF)

    return CONF[dim][0], CONF[dim][1]

#------------------------------------------------------------------------------#
class WeightedRandomGenerator(object):
    """!
    @ingroup Sampling
    Defines a class of weights to be used to select based on linear weighting.
    This can be on index or some form of ordinal ranking.
    """

    def __init__(self, weights):
        """!
        WeightedRandomGenerator class constructor.

        @param self: <em> pointer </em> \n
            The WeightedRandomGenerator pointer. \n
        @param weights: \e array \n
            The array of weights (Higher = more likely to be selected) \n
        """

        ## @var totals
        # <em> list or numpy array: </em> The ordinal ranking
        # or data that is used to generate tehe weights.
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        """!
        Gets the next weight.

        @param self: <em> pointer </em> \n
            The WeightedRandomGenerator pointer. \n

        @return \e integer: The randomly selected index of the weights array. \n
        """
        rnd = rand() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        """!
        Gets the next weight.

        @param self: <em> pointer </em> \n
            The WeightedRandomGenerator pointer. \n

        @return \e integer: The randomly selected index of the weights array. \n
        """
        return self.next()
