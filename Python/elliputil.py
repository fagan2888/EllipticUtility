'''
------------------------------------------------------------------------
This script performs the computations and generates the figures used in
"Advantages of an Ellipse when Modeling Leisure Utility"

This Python script contains the following functions:
    gen_ellip()
    gen_uprightquad()
    MU_sumsq()
    LC_EulSolve()
    LC_c1solve()

This module is compatible with Python 3.5+ and is NOT NECESSARILY
backward compatible with Python 2.x
------------------------------------------------------------------------
'''
# Import packages
import time
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.patches import Ellipse


def gen_ellip(h, k, a, b, mu, graph=True):
    '''
    --------------------------------------------------------------------
    This plots the general functional form of an ellipse and highlights
    the upper-right quadrant.

    [([x - h] / a) ** mu] + [([y - k] / b) ** mu] = 1
    --------------------------------------------------------------------
    INPUTS:
    h     = scalar, x-coordinate of centroid (h, k)
    k     = scalar, y-coordinate of centroid (h, k)
    a     = scalar > 0, horizontal radius of ellipse
    b     = scalar > 0, vertical radius of ellipse
    mu    = scalar > 0, curvature parameter of ellipse
    graph = boolean, =True if graph output

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    N    = integer > 0, number of elements in the support of x
    xvec = (N,) vector, support of x variable
    yvec = (N,) vector, values of y corresponding to the upper-right
           quadrant values of the ellipse from xvec

    RETURNS: xvec, yvec
    --------------------------------------------------------------------
    '''
    N = 1000
    xvec = np.linspace(h, h+a, N)
    yvec = b * ((1 - (((xvec - h) / a) ** mu)) ** (1 / mu)) + k
    if graph:
        e1 = Ellipse((h, k), 2*a, 2*b, 360.0, linewidth=2.0, fill=False,
                     label='Full ellipse')
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_patch(e1)
        plt.plot(xvec, yvec, color='r', linewidth=4,
                 label='Upper-right quadrant')
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator   = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim((h-1.6*a, h+1.6*a))
        plt.ylim((k-1.4*b, k+1.4*b))
        # plt.legend(loc='upper right')
        figname = "images/EllipseGen"
        plt.savefig(figname)
        print("Saved figure: " + figname)
        plt.close()
        # plt.show()

    return xvec, yvec


def gen_uprightquad(hvec, kvec, avec, bvec, muvec, graph=True):
    '''
    --------------------------------------------------------------------
    This plots only the upper-right quadrant of potentially multiple
    ellipses of the form:

    [([x - h] / a) ** mu] + [([y - k] / b) ** mu] = 1
    --------------------------------------------------------------------
    INPUTS:
    hvec  = (I,) vector, x-coordinate of centroid (h, k) of each of I
            ellipses
    kvec  = (I,) vector, y-coordinate of centroid (h, k) of each of I
            ellipses
    avec  = (I,) vector > 0, horizontal radius of ellipse for each of I
            ellipses
    bvec  = (I,) vector > 0, vertical radius of ellipse for each of I
            ellipses
    muvec = (I,) vector > 0, curvature parameter of ellipse for each of
            I ellipses
    graph = boolean, =True if graph output

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    I    = integer >= 1, number of ellipses to produce
    N    = integer >= 1, number of elements (many) in the support of x
    xmat = (I, N) matrix, support of x variable for I ellipses
    ymat = (I, N) matrix, values of y corresponding to the upper-right
           quadrant values of I ellipses from xmat

    RETURNS: xmat, ymat
    --------------------------------------------------------------------
    '''
    I = len(hvec)
    N = 1000
    xmat = np.zeros((I, N))
    ymat = np.zeros((I, N))
    for i in range(I):
        xmat[i, :] = np.linspace(hvec[i], hvec[i] + avec[i], N)
        ymat[i, :] = (bvec[i] *
            ((1 - (((xmat[i, :] - hvec[i]) / avec[i]) ** muvec[i])) **
            (1 / muvec[i])) + kvec[i])
    if graph:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        for i in range(I):
            plt.plot(xmat[i, :], ymat[i, :], linewidth=4,
                label='Ellipse '+str(i+1))
        # for the minor ticks, use no labels; default NullFormatter
        minorLocator   = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.xlim((h-1.6*a, h+1.6*a))
        plt.ylim((k-1.4*b, k+1.4*b))
        plt.legend(loc='upper right')
        figname = "images/UpRightQuadEllipses"
        plt.savefig(figname)
        print("Saved figure: " + figname)
        plt.close()
        # plt.show()

    return xmat, ymat


def MU_sumsq(paramvec, *args):
    '''
    --------------------------------------------------------------------
    This function calculates the sum of squared errors between two
    marginal utility of leisure functions across the support of leisure
    --------------------------------------------------------------------
    INPUTS:
    paramvec    = (2,) vector, level parameter and shape parameter for
                  marginal utility function being fit
    args        = length 3 tuple (lev_base, shape_base, pair)

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    lev_fit    = scalar > 0, level parameter for function being fit
    shape_fit  = scalar > 0, shape parameter for function being fit
    lev_base   = scalar > 0, level parameter for baseline function
    shape_base = scalar > 0, shape parameter for baseline function
    pair       = string, ='CRRAtoCFE' if matching CRRA to CFE,
                 ='ELLIPtoCFE' if matching Elliptical utility to CFE,
                 ='ELLIPtoCRRA' if matching Elliptical utility to CRRA
    leis_min   = scalar > 0, minimum value in support of leisure to use
                 in estimation
    leis_max   = scalar > leis_min, maximum value in support of leisure
                 to use in estimation
    leis_n     = integer >= 2, number of points to use in support of
                 leisure
    leis       = (leis_n,) vector, support of leisure between leis_min
                 and leis_max
    MU_base    = (leis_n,) vector, marginal utility of baseline function
                 over support of leisure
    MU_fit     = (leis_n,) vector, marginal utility of fitting function
                 over support of leisure
    sumsq      = scalar > 0, sum of squared errors between fitted and
                 baseline marginal utility functions

    RETURNS: sumsq
    --------------------------------------------------------------------
    '''
    lev_fit, shape_fit = paramvec
    lev_base, shape_base, pair = args

    if pair == 'CRRAtoCFE':
        leis_min = 0.2
        leis_max = 0.9
        leis_n = 1000
        leis = np.linspace(leis_min, leis_max, leis_n)
        MU_base = lev_base * ((1 - leis) ** (1 / shape_base))
        MU_fit = lev_fit * (leis ** (-shape_fit))
    elif pair == 'ELLIPtoCFE':
        leis_min = 0.15
        leis_max = 0.95
        leis_n = 1000
        leis = np.linspace(leis_min, leis_max, leis_n)
        MU_base = lev_base * ((1 - leis) ** (1 / shape_base))
        MU_fit = lev_fit * (((1 - leis) ** (shape_fit - 1)) *
                 ((1 - ((1 - leis) ** shape_fit)) **
                 ((1 - shape_fit) / shape_fit)))
    elif pair == 'ELLIPtoCRRA':
        leis_min = 0.2
        leis_max = 0.95
        leis_n = 1000
        leis = np.linspace(leis_min, leis_max, leis_n)
        MU_base = lev_base * (leis ** (-shape_base))
        MU_fit = lev_fit * (((1 - leis) ** (shape_fit - 1)) *
                 ((1 - ((1 - leis) ** shape_fit)) **
                 ((1 - shape_fit) / shape_fit)))
    sumsq = ((MU_fit - MU_base) ** 2).sum()

    return sumsq


def LC_EulSolve(nbvec, *args):
    '''
    --------------------------------------------------------------------
    This function returns a vector of Euler errors corresponding to a
    vector of values for n_s and b_{s+1} in a life cycle model
    --------------------------------------------------------------------
    INPUTS:
    nbvec       = (2*S-1,) vector, given values for n_s and b_{s+1}
    args        = length 10 tuple, (S, wbar, rbar, beta, gamma, e_s,
                  lev, curv, util, constrained)
    S           = integer > 1, number of periods an individual lives
    wbar        = scalar > 0, constant real wage
    rbar        = scalar > 0, constant real interest rate
    beta        = scalar in (0, 1), discount factor
    gamma       = scalar >= 1, coefficient of relative risk aversion on
                  consumption
    e_s         = (S,) vector, individual ability levels by age
    lev         = scalar > 0, level parameter (analogous to chi) for
                  given specification for utility of leisure
    curv        = scalar, curvature parameter (eta, theta, or mu) for
                  given specification for utility of leisure
    util        = string, indicates which utility of leisure function to
                  evaluate ("CRRA", "CFE", "Ellip")
    constrained = boolean, =True if optimization accounts for
                  constraints, =False if ignores constraints

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    nvec       = (S,) vector, given values for labor supply by age n_s
    bvec       = (S-1,) vector, given values savings by age b_{s+1}
    b_s        = (S,) vector, savings values from b_1 to b_S, with b_1=0
    b_sp1      = (S,) vector, savings values from b_2 to b_{S+1}, with
                 b_{S+1}=0
    c_s        = (S,) vector, consumption by age from budget constraint
    c_leq0     = (S,) boolean vector, =True for elements where c_s<=0
    MUc_s      = (S,) vector, marginal utility of consumption by age
    lambda2vec = (S,) vector, multipliers by age on n_s>=0 constraint
    lambda3vec = (S,) vector, multipliers by age on n_s<=1 constraint
    n_lt0      = (S,) boolean vector, =True for elements where n_s<0
    n_gt1      = (S,) boolean vector, =True for elements where n_s>1
    n_unc      = (S,) boolean vector, =True for elements where
                 n_s in [0,1] unconstrained
    vprime     = (S,) vector, marginal utility of leisure v'(l)
    penalty    = (N,) vector, penalty function for areas of support
                 outside of the inequality constraints in direction for
                 which there is no Inada condition
    ndiff      = (S,) vector, Euler errors for labor supply equations
    bdiff      = (S-1,) vector, Euler errors for savings equations
    diff       = (2*S-1,) vector, all Euler errors [ndiff, bdiff]

    RETURNS: diff
    --------------------------------------------------------------------
    '''
    S, wbar, rbar, beta, gamma, e_s, lev, curv, util, constrained = args
    nvec = nbvec[:S]
    bvec = nbvec[S:]
    b_s = np.hstack((0, bvec))
    b_sp1 = np.hstack((bvec, 0))
    c_s = (1 + rbar) * b_s + wbar * e_s * nvec - b_sp1
    c_leq0 = c_s <= 0
    MUc_s = np.zeros(S)
    MUc_s[c_leq0==False] = c_s[c_leq0==False] ** (-gamma)
    MUc_s[c_leq0==True] = 10 ** (12)

    if util == 'CRRA':
        lambda2vec = np.zeros(S)
        if constrained:
            n_lt0 = nvec < 0
            n_gt1 = nvec > 1
            n_unc = np.logical_and(n_lt0==False, n_gt1==False)
            vprime = np.zeros(S)
            lambda2vec[n_lt0] = \
                ((lev * (np.ones(n_lt0.sum()) ** (-curv))) -
                (wbar * e_s[n_lt0] * (((1 + rbar) * b_s[n_lt0] -
                b_sp1[n_lt0]) ** (-gamma))))
            penalty = 1.0 * (np.absolute(nvec[n_lt0]) ** 2)
            vprime[n_lt0] = ((1 - nvec[n_lt0]) ** (-curv)) + penalty
            vprime[n_gt1] = 10 ** (13)
            vprime[n_unc] = (1 - nvec[n_unc]) ** (-curv)
        else:
            vprime = (1 - nvec) ** (-curv)
        ndiff = wbar * e_s * MUc_s - lev * vprime + lambda2vec
    elif util == 'CFE':
        lambda3vec = np.zeros(S)
        if constrained:
            n_lt0 = nvec < 0
            n_gt1 = nvec > 1
            n_unc = np.logical_and(n_lt0==False, n_gt1==False)
            vprime = np.zeros(S)
            lambda3vec[n_gt1] = (wbar * e_s[n_gt1] *
                (((1 + rbar) * b_s[n_gt1] + wbar * e_s[n_gt1] -
                b_sp1[n_gt1]) ** (-gamma)) -
                lev * (np.ones(n_gt1.sum()) ** (1 / curv)))
            penalty = 10 * ((nvec[n_gt1] - 1) ** 3)
            vprime[n_gt1] = nvec[n_gt1] ** (1 / curv) - penalty
            vprime[n_lt0] = -10 * (nvec[n_lt0] ** 2)
            vprime[n_unc] = nvec[n_unc] ** (1 / curv)
        else:
            vprime = nvec ** (1 / curv)
        ndiff = wbar * e_s * MUc_s - lev * vprime - lambda3vec
    elif util == 'Ellip':
        n_lt0 = nvec < 0
        n_gt1 = nvec > 1
        n_unc = np.logical_and(n_lt0==False, n_gt1==False)
        vprime = np.zeros(S)
        vprime[n_lt0] = -10 * (nvec[n_lt0] ** 2)
        vprime[n_gt1] = 10 ** (14)
        vprime[n_unc] = ((nvec[n_unc] ** (curv - 1)) *
            ((1 - (nvec[n_unc] ** curv)) ** ((1 - curv) / curv)))
        ndiff = wbar * e_s * MUc_s - lev * vprime
    bdiff = MUc_s[:-1] - beta * (1 + rbar) * MUc_s[1:]
    diff = np.hstack((ndiff, bdiff))

    return diff


def LC_c1solve(c1, *args):
    '''
    --------------------------------------------------------------------
    This function returns the scalar value of final period savings
    b_{S+1}, which should equal zero, as a function of the value of
    first-period consumption c1 is a life cycle model. All other
    variables are functions of c1 in this simple model
    --------------------------------------------------------------------
    INPUTS:
    c1    = scalar > 0, value for initial age consumption
    args  = length 9 tuple,
            (S, wbar, rbar, beta, gamma, e_s, lev, curv, util)
    S     = integer > 1, number of periods an individual lives
    wbar  = scalar > 0, constant real wage
    rbar  = scalar > 0, constant real interest rate
    beta  = scalar in (0, 1), discount factor
    gamma = scalar >= 1, coefficient of relative risk aversion on
            consumption
    e_s   = (S,) vector, individual ability levels by age
    lev   = scalar > 0, level parameter (analogous to chi) for
            given specification for utility of leisure
    curv  = scalar, curvature parameter (eta, theta, or mu) for
            given specification for utility of leisure
    util  = string, indicates which utility of leisure function to
            evaluate ("CRRA", "CFE", "Ellip")

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    c_s    = (S,) vector, consumption by age
    MUc_s  = (S,) vector, marginal utility of consumption by age
    n_unc  = (S,) vector, labor supply by age implied by Euler equation
             without Lagrange multipliers (unconstrained)
    n_s    = (S,) vector, labor supply by age where values are
             constrained to be between 0 and 1
    b1     = scalar, initial-age wealth/savings (equals 0)
    b_s    = (S+1,) vector, lifetime wealth from age 1 to S+1
    b_last = scalar, savings for age S+1 (in equilibrium is 0)

    RETURNS: b_last
    --------------------------------------------------------------------
    '''
    S, wbar, rbar, beta, gamma, e_s, lev, curv, util = args
    c_s = np.zeros(S)
    c_s[0] = c1
    for age in range(1, S):
        c_s[age] = c_s[age-1] * ((beta * (1 + rbar)) ** (1 / gamma))
    MUc_s = c_s ** (-gamma)
    if util == "CRRA":
        n_unc = 1 - ((wbar * e_s * MUc_s) / lev) ** (-1 / curv)
    elif util == "CFE":
        n_unc = ((wbar * e_s * MUc_s) / lev) ** curv
    # "Ellip" requires a root finder because is has no closed-form
    # solution
    # elif util == "Ellip":
    #     n_unc =
    n_s = np.minimum(1, np.maximum(0, n_unc))
    b1 = 0
    b_s = np.zeros(S+1)
    b_s[0] = b1
    for age in range(1, S+1):
        b_s[age] = ((1 + rbar) * b_s[age-1] +
                   wbar * e_s[age-1] * n_s[age-1] - c_s[age-1])
    b_last = b_s[-1]

    return b_last


'''
------------------------------------------------------------------------
Generate figure of general ellipse for Figure 1 in paper
[h, k, a, b, mu] = [1, -1, 1, 2, 2] where
[([x - h] / a) ** mu] + [([y - k] / b) ** mu] = 1
------------------------------------------------------------------------
h1    = scalar, x-coordinate of centroid of ellipse
k1    = scalar, y-coordinate of centroid of ellipse
a1    = scalar > 0, ellipse radius in x-direction
b1    = scalar > 0, ellipse radius in y-direction
mu1   = scalar >=1, curvature parameter of ellipse
xvec1 = (1000,) vector, support of x for upper-right quadrant of ellipse
yvec1 = (1000,) vector, y values corresponding to xvec1, upper-right
        quadrant of ellipse
------------------------------------------------------------------------
'''
h1 = 1.0
k1 = -1.0
a1 = 1.0
b1 = 2.0
mu1 = 2.0
xvec1, yvec1 = gen_ellip(h1, k1, a1, b1, mu1, True)


'''
------------------------------------------------------------------------
Generate Figure 2 of specific ellipse with a,b=1, h,k=0 and three
values of mu = 1,2,3
------------------------------------------------------------------------
hvec2  = (3,) vector, values for h in three different ellipses
kvec2  = (3,) vector, values for k in three different ellipses
avec2  = (3,) vector, values for a in three different ellipses
bvec2  = (3,) vector, values for b in three different ellipses
mu1    = scalar >= 1, value for mu in first ellipse
mu2    = scalar >= 1, value for mu in second ellipse
mu3    = scalar >= 1, value for mu in third ellipse
muvec2 = (3,) vector, values for mu in three different ellipses
xmat2  = (3, 1000) matrix, support of x for upper-right quadrant of
         three ellipses
ymat2  = (3, 1000) matrix, y values corresponding to xmat2, upper-right
         quadrant of three ellipses
------------------------------------------------------------------------
'''
hvec2 = np.array([0.0, 0.0, 0.0])
kvec2 = np.array([0.0, 0.0, 0.0])
avec2 = np.array([1.0, 1.0, 1.0])
bvec2 = np.array([1.0, 1.0, 1.0])
mu1 = 1.0
mu2 = 2.0
mu3 = 3.0
muvec2 = np.array([mu1, mu2, mu3])
xmat2, ymat2 = gen_uprightquad(hvec2, kvec2, avec2, bvec2, muvec2,
               False)
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.plot(xmat2[0, :], ymat2[0, :], '-', linewidth=4,
        label='$\mu$='+str(mu1))
plt.plot(xmat2[1, :], ymat2[1, :], '--', linewidth=4,
        label='$\mu$='+str(mu2))
plt.plot(xmat2[2, :], ymat2[2, :], ':', linewidth=4,
        label='$\mu$='+str(mu3))
# for the minor ticks, use no labels; default NullFormatter
minorLocator   = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.xlabel(r'Labor supply $n=1-\ell$')
plt.ylabel(r'utility of leisure $v(\ell)$')
plt.xlim((-0.15, 1.15))
plt.ylim((-0.15, 1.15))
plt.legend(loc='lower left')
figname = "images/UpRightQuadEllipses"
plt.savefig(figname)
print("Saved figure: " + figname)
plt.close()
# plt.show()


'''
------------------------------------------------------------------------
Fit CRRA utility function to baseline CFE utility function by matching
marginal utilities along the support of leisure
------------------------------------------------------------------------
chi_CFE          = scalar > 0, level parameter for disultility of labor
                   in CFE function
theta            = scalar > 0, curvature parameter for disutility of
                   labor in CFE function, constant Frisch elasticity
chi_CRRA_init    = scalar > 0, initial guess for level parameter in CRRA
                   utility of leisure function
eta_init         = scalar > 0, initial guess for curvature parameter in
                   CRRA utility of leisure function
pair_CRRA        = string, tells MU_sumsq function which utility
                   function is being fit to which
bnds_CRRA        = length 2 tuple, lower and upper bounds for parameters
                   being optimized
CRRA_params_init = length 2 tuple, initial guesses for parameters being
                   optimized
CRRA_fit_args    = length 3 tuple, arguments to be passed into MU_sumsq
CRRA_opt_args    = length 9 tuple, output from constrained optimization
chi_CRRA         = scalar > 0, fitted value of level parameter for CRRA
                   utility of leisure function
eta              = scalar > 0, fitted value of curvature parameter for
                   CRRA utility of leisure function
SSE_CRRA         = scalar > 0, optimized sum of squared errors
leisure          = (500,) vector, values in support of leisure
MU_CFE           = (500,) vector, values of marginal utility of CFE
                   function given leisure
MU_CRRA          = (500,) vector, values of marginal utility of CRRA
                   function given leisure
------------------------------------------------------------------------
'''
chi_CFE = 1.0
theta = 0.5
chi_CRRA_init = 1.0
eta_init = 2.5
pair_CRRA = 'CRRAtoCFE'
bnds_CRRA = ((1e-12, None), (1e-12, None))
CRRA_params_init = (chi_CRRA_init, eta_init)
CRRA_fit_args = (chi_CFE, theta, pair_CRRA)
CRRA_opt_args = opt.minimize(MU_sumsq, CRRA_params_init,
    args=(CRRA_fit_args), method="L-BFGS-B", bounds=bnds_CRRA,
    tol=1e-15)
chi_CRRA, eta = CRRA_opt_args.x
SSE_CRRA = CRRA_opt_args.fun
leisure = np.linspace(0.2, 1.0, 500)
MU_CFE = chi_CFE * ((1 - leisure) ** (1 / theta))
MU_CRRA = chi_CRRA * (leisure ** (-eta))

# Plot MU_CFE and MU_CRRA
fig, ax = plt.subplots()
plt.plot(leisure, MU_CFE, '-', linewidth=4, label='MU CFE')
plt.plot(leisure, MU_CRRA, '--', linewidth=4, label='MU CRRA')
# for the minor ticks, use no labels; default NullFormatter
minorLocator   = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.xlabel(r'Leisure $\ell$')
plt.ylabel(r'marginal utility of leisure $v\'(\ell)$')
plt.xlim((0.1, 1.0))
plt.ylim((-0.05, 0.95))
plt.legend(loc='upper right')
figname = "images/MU_CFE_CRRA"
plt.savefig(figname)
print("Saved figure: " + figname)
plt.close()
# plt.show()


'''
------------------------------------------------------------------------
Fit Elliptical utility function to fitted CRRA utility function by
matching marginal utilities along the support of leisure
------------------------------------------------------------------------
chi_Ellip_init     = scalar > 0, initial guess for level parameter in
                     Elliptical utility of leisure function
mu_init            = scalar > 0, initial guess for curvature parameter
                     in Elliptical utility of leisure function
pair_Ellip_CRRA    = string, tells MU_sumsq function which utility
                     function is being fit to which
bnds_Ellip         = length 2 tuple, lower and upper bounds for
                     parameters being optimized
Ellip_params_init  = length 2 tuple, initial guesses for parameters
                     being optimized
EllipCRRA_fit_args = length 3 tuple, arguments to be passed to MU_sumsq
EllipCRRA_opt_args = length 9 tuple, output from constrained optimizat'n
chi_EllipCRRA      = scalar > 0, fitted value of level parameter for
                     Elliptical utility of leisure function
mu_EllipCRRA       = scalar > 0, fitted value of curvature parameter for
                     Elliptical utility of leisure function
SSE_EllipCRRA      = scalar > 0, optimized sum of squared errors
leisure2           = (500,) vector, values in support of leisure
MU_EllipCRRA       = (500,) vector, values of marginal utility of
                     Elliptical function given leisure
labor2             = (500,) vector, values in support of labor
Frisch_EllipCRRA   = (500,) vector, values of Frisch elasticity of labor
                     of Elliptical function given labor
Frisch_CRRA        = (500,) vector, values of Frisch elasticity of labor
                     of CRRA function given labor
------------------------------------------------------------------------
'''
chi_Ellip_init = 4.0
mu_init = 4.5
pair_Ellip_CRRA = 'ELLIPtoCRRA'
bnds_Ellip = ((1e-12, None), (1e-12, None))
Ellip_params_init = (chi_Ellip_init, mu_init)
EllipCRRA_fit_args = (chi_CRRA, eta, pair_Ellip_CRRA)
EllipCRRA_opt_args = opt.minimize(MU_sumsq, Ellip_params_init,
    args=(EllipCRRA_fit_args), method="L-BFGS-B", bounds=bnds_Ellip,
    tol=1e-15)
chi_EllipCRRA, mu_EllipCRRA = EllipCRRA_opt_args.x
SSE_EllipCRRA = EllipCRRA_opt_args.fun
leisure2 = np.linspace(0.15, 1.0, 500)
MU_EllipCRRA = chi_EllipCRRA * (((1 - leisure2) ** (mu_EllipCRRA - 1)) *
               ((1 - ((1 - leisure2) ** mu_EllipCRRA)) **
               ((1 - mu_EllipCRRA) / mu_EllipCRRA)))
labor2 = np.linspace(0.2, 1.0, 500)
Frisch_EllipCRRA = (1 - (labor2 ** mu_EllipCRRA)) / (mu_EllipCRRA - 1)
Frisch_CRRA = (1 / eta) * ((1 - labor2) / labor2)

# Plot MU_Ellip against MU_CRRA
fig, ax = plt.subplots()
plt.plot(leisure2, MU_CRRA, '-', linewidth=4, label='MU CRRA')
plt.plot(leisure2, MU_EllipCRRA, '--', linewidth=4, label='MU Ellipse')
# for the minor ticks, use no labels; default NullFormatter
minorLocator   = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.xlabel(r'Leisure $\ell$')
plt.ylabel(r'marginal utility of leisure $v\'(\ell)$')
plt.xlim((0.1, 1.0))
plt.ylim((-0.05, 0.95))
plt.legend(loc='upper right')
figname = "images/MU_Ellip_CRRA"
plt.savefig(figname)
print("Saved figure: " + figname)
plt.close()
# plt.show()

# Plot Frisch_Ellip against Frisch_CRRA
fig, ax = plt.subplots()
plt.plot(labor2, Frisch_CRRA, '-', linewidth=4,
        label='Frisch $\epsilon$ CRRA')
plt.plot(labor2, Frisch_EllipCRRA, '--', linewidth=4,
        label='Frisch $\epsilon$ Ellipse')
# for the minor ticks, use no labels; default NullFormatter
minorLocator   = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.xlabel(r'Labor $n$')
plt.ylabel(r'Frisch elasticity $\epsilon$')
# plt.xlim((0.1, 1.0))
# plt.ylim((-0.05, 0.95))
plt.legend(loc='upper right')
figname = "images/Frisch_Ellip_CRRA"
plt.savefig(figname)
print("Saved figure: " + figname)
plt.close()
# plt.show()


'''
------------------------------------------------------------------------
Fit Elliptical utility function to baseline CFE utility function by
matching marginal utilities along the support of leisure
------------------------------------------------------------------------
pair_Ellip_CFE    = string, tells MU_sumsq function which utility
                    function is being fit to which
EllipCFE_fit_args = length 3 tuple, arguments to be passed to MU_sumsq
EllipCFE_opt_args = length 9 tuple, output from constrained optimization
chi_EllipCFE      = scalar > 0, fitted value of level parameter for
                    Elliptical utility of leisure function
mu_EllipCFE       = scalar > 0, fitted value of curvature parameter for
                    Elliptical utility of leisure function
SSE_EllipCFE      = scalar > 0, optimized sum of squared errors
MU_EllipCFE       = (500,) vector, values of marginal utility of
                    Elliptical function given leisure
labor3            = (500,) vector, values in support of labor
Frisch_EllipCFE   = (500,) vector, values of Frisch elasticity of labor
                    of Elliptical function given labor
Frisch_CFE        = (500,) vector, values of Frisch elasticity of labor
                    of CFE function given labor
------------------------------------------------------------------------
'''
pair_Ellip_CFE = 'ELLIPtoCFE'
EllipCFE_fit_args = (chi_CFE, theta, pair_Ellip_CFE)
EllipCFE_opt_args = opt.minimize(MU_sumsq, Ellip_params_init,
    args=(EllipCFE_fit_args), method="L-BFGS-B", bounds=bnds_Ellip,
    tol=1e-15)
chi_EllipCFE, mu_EllipCFE = EllipCFE_opt_args.x
SSE_EllipCFE = EllipCFE_opt_args.fun
MU_EllipCFE = chi_EllipCFE * (((1 - leisure2) ** (mu_EllipCFE - 1)) *
               ((1 - ((1 - leisure2) ** mu_EllipCFE)) **
               ((1 - mu_EllipCFE) / mu_EllipCFE)))
labor3 = np.linspace(0.0, 1.0, 500)
Frisch_EllipCFE = (1 - (labor3 ** mu_EllipCFE)) / (mu_EllipCFE - 1)
Frisch_CFE = theta * np.ones(labor3.shape[0])

# Plot MU_Ellip against MU_CFE
fig, ax = plt.subplots()
plt.plot(leisure2, MU_CFE, '-', linewidth=4, label='MU CFE')
plt.plot(leisure2, MU_EllipCFE, '--', linewidth=4, label='MU Ellipse')
# for the minor ticks, use no labels; default NullFormatter
minorLocator   = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.xlabel(r'Leisure $\ell$')
plt.ylabel(r'marginal utility of leisure $v\'(\ell)$')
# plt.xlim((0.1, 1.0))
# plt.ylim((-0.05, 0.95))
plt.legend(loc='upper right')
figname = "images/MU_Ellip_CFE"
plt.savefig(figname)
print("Saved figure: " + figname)
plt.close()
# plt.show()

# Plot Frisch_Ellip against Frisch_CFE
fig, ax = plt.subplots()
plt.plot(labor3, Frisch_CFE, '-', linewidth=4,
         label='Frisch $\epsilon$ CFE')
plt.plot(labor3, Frisch_EllipCFE, '--', linewidth=4,
         label='Frisch $\epsilon$ Ellipse')
# for the minor ticks, use no labels; default NullFormatter
minorLocator   = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
plt.grid(b=True, which='major', color='0.65',linestyle='-')
plt.xlabel(r'Labor $n$')
plt.ylabel(r'Frisch elasticity $\epsilon$')
# plt.xlim((0.1, 1.0))
# plt.ylim((-0.05, 0.95))
plt.legend(loc='upper right')
figname = "images/Frisch_Ellip_CFE"
plt.savefig(figname)
print("Saved figure: " + figname)
plt.close()
# plt.show()


'''
------------------------------------------------------------------------
Simulate partial equilibrium life cycle model under the CRRA, CFE, and
Elliptical utility functions from Table 1 of the paper.
------------------------------------------------------------------------
plot_es           = boolean, =True generates a plot for ability
plot_LC_CRRA1     = boolean, =True generates a plot for life cycle model
                    outcomes with CRRA utility of leisure unconstrained
plot_LC_CRRA2     = boolean, =True generates a plot for life cycle model
                    outcomes with CRRA utility of leisure constrained
plot_LC_CFE1      = boolean, =True generates a plot for life cycle model
                    outcomes with CFE utility of leisure unconstrained
plot_LC_CFE2      = boolean, =True generates a plot for life cycle model
                    outcomes with CFE utility of leisure constrained
plot_LC_EllipCRRA = boolean, =True generates a plot for life cycle model
                    outcomes with Elliptical utility of leisure fitted
                    to CRRA
plot_LC_EllipCFE  = boolean, =True generates a plot for life cycle model
                    outcomes with Elliptical utility of leisure fitted
                    to CFE
S                 = integer >= 2, number of periods an individual lives
                    (must be hard coded to 20 for this)
gamma             = scalar >= 1, coefficient of relative risk aversion
                    on utility of consumption
rbar_an           = scalar > 0, annual real interest rate
rbar              = scalar > 0, model period real interest rate
wbar              = scalar > 0, model period real wage
e_s               = (20,) vector, ability levels by age
beta              = scalar in (0, 1), discount factor
------------------------------------------------------------------------
'''
plot_es = True
plot_LC_CRRA1 = True
plot_LC_CRRA2 = True
plot_LC_CFE1 = True
plot_LC_CFE2 = True
plot_LC_EllipCRRA = True
plot_LC_EllipCFE = True
S = 20
gamma = 2.2
rbar_an = 0.05
rbar = ((1 + rbar_an) ** (80/S)) - 1
wbar = 1
e_s = np.array([0.45473894, 0.57630271, 0.71985346, 0.87613283,
        1.03581699, 1.1866692, 1.31572706, 1.4081917, 1.45227009,
        1.44134383, 1.37467649, 1.25940399, 1.10416403, 0.92300394,
        0.7302676, 0.54861603, 0.37617533, 0.21963054, 0.1017821,
        0.03411342])
beta = 1 / (1 + rbar)

if plot_es:
    # Plot e_s vector
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), e_s, '-', linewidth=4)
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'ability level $e_s$')
    plt.xlim((0, S+1))
    # plt.ylim((-0.05, 0.95))
    figname = "images/abilities"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()


'''
------------------------------------------------------------------------
Solve for optimal CRRA utility of leisure v(l) solution in life cycle
model when l is unconstrained
------------------------------------------------------------------------
wbar_CRRA          = scalar > 0, real wage
util_CRRA          = string, "CRRA" denotes the specification of leisure
                     utility
cnstr_CRRA1        = boolean, =True means the problem is constrained
n_guess            = (S,) vector, initial guess for labor supply
b_guess            = (S-1,) vector, initial guess for savings
guesses            = (2*S-1,) vector, initial guesses [n_guess, b_guess]
LC_CRRA1_args      = length 10 tuple, arguments to be passed into solver
start_time_CRRA1   = scalar > 0, representation of computation start time
LC_CRRA1_output    = length 10 dictionary, output objects from root
                     finder
elapsed_time_CRRA1 = scalar, elapsed time (in seconds) for computation
n_CRRA1            = (S,) vector, optimal labor supply over life cycle
b_CRRA1            = (S-1,) vector, optimal savings over life cycle
b_s_CRRA1          = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [1, S] with b1 = 0
b_sp1_CRRA1        = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [2, S+1] with b_{S+1} = 0
c_CRRA1            = (S,) vector, optimal consumption over life cycle
------------------------------------------------------------------------
'''
wbar_CRRA = 3
util_CRRA = "CRRA"
cnstr_CRRA1 = False
n_guess = 0.9 * np.ones(S)
b_guess = 0.01 * np.ones(S-1)
guesses = np.hstack((n_guess, b_guess))
LC_CRRA1_args = (S, wbar_CRRA, rbar, beta, gamma, e_s, chi_CRRA, eta,
                util_CRRA, cnstr_CRRA1)
start_time_CRRA1 = time.clock()
LC_CRRA1_output = opt.root(LC_EulSolve, guesses, args=(LC_CRRA1_args),
                  method='lm', tol=1e-14)
elapsed_time_CRRA1 = time.clock() - start_time_CRRA1
n_CRRA1 = LC_CRRA1_output.x[:S]
b_CRRA1 = LC_CRRA1_output.x[S:]
b_s_CRRA1 = np.hstack((0, b_CRRA1))
b_sp1_CRRA1 = np.hstack((b_CRRA1, 0))
c_CRRA1 = ((1 + rbar) * b_s_CRRA1 + wbar_CRRA * e_s * n_CRRA1 -
          b_sp1_CRRA1)

if plot_LC_CRRA1:
    # CRRA1 values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_CRRA1, '-', linewidth=4,
            label='labor supply')
    plt.plot(np.linspace(1, S, S), b_sp1_CRRA1, '--', linewidth=4,
            label='savings')
    plt.plot(np.linspace(1, S, S), c_CRRA1, ':', linewidth=4,
            label='consumption')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'optimal values $n_s$, $b_{s+1}$, $c_s$')
    plt.xlim((0, S+1))
    plt.ylim((-5.0, 5.0))
    plt.legend(loc='upper left')
    figname = "images/LC_CRRA1"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()


'''
------------------------------------------------------------------------
Solve for optimal CRRA utility of leisure v(l) solution in life cycle
model when l is constrained
------------------------------------------------------------------------
cnstr_CRRA2        = boolean, =True means the problem is constrained
LC_CRRA2_args      = length 10 tuple, arguments to be passed into solver
start_time_CRRA2   = scalar > 0, representation of computation start time
LC_CRRA2_output    = length 10 dictionary, output objects from root
                     finder
elapsed_time_CRRA2 = scalar, elapsed time (in seconds) for computation
n_CRRA2            = (S,) vector, optimal labor supply over life cycle
b_CRRA2            = (S-1,) vector, optimal savings over life cycle
b_s_CRRA2          = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [1, S] with b1 = 0
b_sp1_CRRA2        = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [2, S+1] with b_{S+1} = 0
c_CRRA2            = (S,) vector, optimal consumption over life cycle
------------------------------------------------------------------------
'''
cnstr_CRRA2 = True
LC_CRRA2_args = (S, wbar_CRRA, rbar, beta, gamma, e_s, chi_CRRA, eta,
                util_CRRA, cnstr_CRRA2)
start_time_CRRA2 = time.clock()
LC_CRRA2_output = opt.root(LC_EulSolve, guesses, args=(LC_CRRA2_args),
                  method='lm', tol=1e-14)
elapsed_time_CRRA2 = time.clock() - start_time_CRRA2
n_CRRA2 = LC_CRRA2_output.x[:S]
b_CRRA2 = LC_CRRA2_output.x[S:]
b_s_CRRA2 = np.hstack((0, b_CRRA2))
b_sp1_CRRA2 = np.hstack((b_CRRA2, 0))
c_CRRA2 = ((1 + rbar) * b_s_CRRA2 + wbar_CRRA * e_s * n_CRRA2 -
          b_sp1_CRRA2)

if plot_LC_CRRA2:
    # CRRA2 values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_CRRA2, '-', linewidth=4,
            label='labor supply')
    plt.plot(np.linspace(1, S, S), b_sp1_CRRA2, '--', linewidth=4,
            label='savings')
    plt.plot(np.linspace(1, S, S), c_CRRA2, ':', linewidth=4,
            label='consumption')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'optimal values $n_s$, $b_{s+1}$, $c_s$')
    plt.xlim((0, S+1))
    plt.ylim((-5.0, 5.0))
    plt.legend(loc='upper left')
    figname = "images/LC_CRRA2"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()


'''
------------------------------------------------------------------------
Solve for optimal CFE utility of leisure v(l) solution in life cycle
model when l is unconstrained
------------------------------------------------------------------------
wbar_CFE          = scalar > 0, real wage
util_CFE          = string, "CFE" denotes the specification of leisure
                    utility
cnstr_CFE1        = boolean, =True means the problem is constrained
LC_CFE1_args      = length 10 tuple, arguments to be passed into solver
start_time_CFE1   = scalar > 0, representation of computation start time
LC_CFE1_output    = length 10 dictionary, output objects from root
                    finder
elapsed_time_CFE1 = scalar, elapsed time (in seconds) for computation
n_CFE1            = (S,) vector, optimal labor supply over life cycle
b_CFE1            = (S-1,) vector, optimal savings over life cycle
b_s_CFE1          = (S,) vector, optimal savings/wealth over life cycle
                    for ages s in [1, S] with b1 = 0
b_sp1_CFE1        = (S,) vector, optimal savings/wealth over life cycle
                    for ages s in [2, S+1] with b_{S+1} = 0
c_CFE1            = (S,) vector, optimal consumption over life cycle
------------------------------------------------------------------------
'''
wbar_CFE = 1.5
util_CFE = "CFE"
cnstr_CFE1 = False
LC_CFE1_args = (S, wbar_CFE, rbar, beta, gamma, e_s, chi_CFE, theta,
               util_CFE, cnstr_CFE1)
start_time_CFE1 = time.clock()
LC_CFE1_output = opt.root(LC_EulSolve, guesses, args=(LC_CFE1_args),
                  method='lm', tol=1e-14)
elapsed_time_CFE1 = time.clock() - start_time_CFE1
n_CFE1 = LC_CFE1_output.x[:S]
b_CFE1 = LC_CFE1_output.x[S:]
b_s_CFE1 = np.hstack((0, b_CFE1))
b_sp1_CFE1 = np.hstack((b_CFE1, 0))
c_CFE1 = (1 + rbar) * b_s_CFE1 + wbar_CFE * e_s * n_CFE1 - b_sp1_CFE1

if plot_LC_CFE1:
    # CFE1 values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_CFE1, '-', linewidth=4,
            label='labor supply')
    plt.plot(np.linspace(1, S, S), b_sp1_CFE1, '--', linewidth=4,
            label='savings')
    plt.plot(np.linspace(1, S, S), c_CFE1, ':', linewidth=4,
            label='consumption')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'optimal values $n_s$, $b_{s+1}$, $c_s$')
    plt.xlim((0, S+1))
    plt.ylim((-4.0, 4.0))
    plt.legend(loc='upper left')
    figname = "images/LC_CFE1"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()


'''
------------------------------------------------------------------------
Solve for optimal CFE utility of leisure v(l) solution in life cycle
model when l is constrained
------------------------------------------------------------------------
cnstr_CFE2         = boolean, =True means the problem is constrained
LC_CFE2_args       = length 10 tuple, arguments to be passed into solver
start_time_CFE2    = scalar > 0, representation of computation start
                     time
LC_CFE2_output     = length 10 dictionary, output objects from root
                     finder
elapsed_time_CFE2  = scalar, elapsed time (in seconds) for computation
n_CFE2             = (S,) vector, optimal labor supply over life cycle
b_CFE2             = (S-1,) vector, optimal savings over life cycle
b_s_CFE2           = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [1, S] with b1 = 0
b_sp1_CFE2         = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [2, S+1] with b_{S+1} = 0
c_CFE2             = (S,) vector, optimal consumption over life cycle
c_init             = scalar > 0, initial guess for c1 in alternative
                     solution method
LC_CFE2b_args      = length 9 tuple, arguments to be passed into solver
start_time_CFE2b   = scalar > 0, representation of computation start
                     time
LC_CFE2b_output    = length 10 dictionary, output objects from root
                     finder
elapsed_time_CFE2b = scalar, elapsed time (in seconds) for computation
c1                 = scalar, optimal first period consumption
c_CFE2b            = (S,) vector, optimal life cycle consumption by age
age                = integer >= 1, index for age
MUc_s              = (S,) vector, marginal utility of consumption by age
n_unc              = (S,) vector, unconstrained labor supply implied by
                     Euler equation
n_CFE2b            = (S,) vector, truncated n_unc at n=0 for n_unc<0 and
                     n=1 for n_unc>1
b1                 = scalar, initial wealth/savings (equals 0)
b_s                = (S+1,) vector, lifetime savings/wealth by age
b_CFE2b            = (S-1,) vector, lifetime savings/wealth by age for
                     s in [2,S]
b_s_CFE2b          = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [1, S] with b1 = 0
b_sp1_CFE2b        = (S,) vector, optimal savings/wealth over life cycle
                     for ages s in [2, S+1] with b_{S+1} = 0
------------------------------------------------------------------------
'''
cnstr_CFE2 = True
LC_CFE2_args = (S, wbar_CFE, rbar, beta, gamma, e_s, chi_CFE, theta,
               util_CFE, cnstr_CFE2)
start_time_CFE2 = time.clock()
LC_CFE2_output = opt.root(LC_EulSolve, guesses, args=(LC_CFE2_args),
                  method='lm', tol=1e-14)
elapsed_time_CFE2 = time.clock() - start_time_CFE2
n_CFE2 = LC_CFE2_output.x[:S]
b_CFE2 = LC_CFE2_output.x[S:]
b_s_CFE2 = np.hstack((0, b_CFE2))
b_sp1_CFE2 = np.hstack((b_CFE2, 0))
c_CFE2 = (1 + rbar) * b_s_CFE2 + wbar_CFE * e_s * n_CFE2 - b_sp1_CFE2
# Alternative computational approach to get it perfect in this simple
# model
c_init = 0.5
LC_CFE2b_args = (S, wbar_CFE, rbar, beta, gamma, e_s, chi_CFE, theta,
                util_CFE)
start_time_CFE2b = time.clock()
LC_CFE2b_output = opt.root(LC_c1solve, c_init, args=(LC_CFE2b_args),
                  method='lm', tol=1e-14)
elapsed_time_CFE2b = time.clock() - start_time_CFE2b
c1 = LC_CFE2b_output.x[0]
c_CFE2b = np.zeros(S)
c_CFE2b[0] = c1
for age in range(1, S):
    c_CFE2b[age] = c_CFE2b[age-1] * ((beta * (1 + rbar)) ** (1 / gamma))
MUc_s = c_CFE2b ** (-gamma)
n_unc = ((wbar_CFE * e_s * MUc_s) / chi_CFE) ** theta
n_CFE2b = np.minimum(1, np.maximum(0, n_unc))
b1 = 0
b_s = np.zeros(S+1)
b_s[0] = b1
for age in range(1, S+1):
    b_s[age] = ((1 + rbar) * b_s[age-1] +
               wbar_CFE * e_s[age-1] * n_CFE2b[age-1] - c_CFE2b[age-1])
b_CFE2b = b_s[1: -1]
b_s_CFE2b = b_s[:-1]
b_sp1_CFE2b = b_s[1:]

if plot_LC_CFE2:
    # CFE2b values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_CFE2b, '-', linewidth=4,
            label='labor supply')
    plt.plot(np.linspace(1, S, S), b_sp1_CFE2b, '--', linewidth=4,
            label='savings')
    plt.plot(np.linspace(1, S, S), c_CFE2b, ':', linewidth=4,
            label='consumption')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'optimal values $n_s$, $b_{s+1}$, $c_s$')
    plt.xlim((0, S+1))
    plt.ylim((-4.0, 4.0))
    plt.legend(loc='upper left')
    figname = "images/LC_CFE2b"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()


'''
------------------------------------------------------------------------
Solve for optimal elliptical utility of leisure v(l) solution in life
cycle model estimated to fit the CRRA specification
------------------------------------------------------------------------
util_Ellip             = string, "Ellip" denotes the specification of
                         leisure utility
cnstr_Ellip            = boolean, =True means the problem is constrained
LC_EllipCRRA_args      = length 10 tuple, arguments to be passed into
                         solver
start_time_EllipCRRA   = scalar > 0, representation of computation start
                         time
LC_EllipCRRA_output    = length 10 dictionary, output objects from root
                         finder
elapsed_time_EllipCRRA = scalar, elapsed time (in seconds) for
                         computation
n_EllipCRRA            = (S,) vector, optimal labor supply over life
                         cycle
b_EllipCRRA            = (S-1,) vector, optimal savings over life cycle
b_s_EllipCRRA          = (S,) vector, optimal savings/wealth over life
                         cycle for ages s in [1, S] with b1 = 0
b_sp1_EllipCRRA        = (S,) vector, optimal savings/wealth over life
                         cycle for ages s in [2, S+1] with b_{S+1} = 0
c_EllipCRRA            = (S,) vector, optimal consumption over life
                         cycle
------------------------------------------------------------------------
'''
util_Ellip = "Ellip"
cnstr_Ellip = False
LC_EllipCRRA_args = (S, wbar_CRRA, rbar, beta, gamma, e_s,
    chi_EllipCRRA, mu_EllipCRRA, util_Ellip, cnstr_Ellip)
start_time_EllipCRRA = time.clock()
LC_EllipCRRA_output = opt.root(LC_EulSolve, guesses,
    args=(LC_EllipCRRA_args), method='lm', tol=1e-14)
elapsed_time_EllipCRRA = time.clock() - start_time_EllipCRRA
n_EllipCRRA = LC_EllipCRRA_output.x[:S]
b_EllipCRRA = LC_EllipCRRA_output.x[S:]
b_s_EllipCRRA = np.hstack((0, b_EllipCRRA))
b_sp1_EllipCRRA = np.hstack((b_EllipCRRA, 0))
c_EllipCRRA = ((1 + rbar) * b_s_EllipCRRA +
    wbar_CRRA * e_s * n_EllipCRRA - b_sp1_EllipCRRA)

if plot_LC_EllipCRRA:
    # EllipCRRA values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_EllipCRRA, '-', linewidth=4,
            label='labor supply')
    plt.plot(np.linspace(1, S, S), b_sp1_EllipCRRA, '--', linewidth=4,
            label='savings')
    plt.plot(np.linspace(1, S, S), c_EllipCRRA, ':', linewidth=4,
            label='consumption')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'optimal values $n_s$, $b_{s+1}$, $c_s$')
    plt.xlim((0, S+1))
    plt.ylim((-5.0, 5.0))
    plt.legend(loc='upper left')
    figname = "images/LC_EllipCRRA"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()

    # Ellip, CRRA1, and CRRA2 labor supply values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_CRRA1, '-', linewidth=4,
            label='unconstrained')
    plt.plot(np.linspace(1, S, S), n_CRRA2, '--', linewidth=4,
            label='constrained')
    plt.plot(np.linspace(1, S, S), n_EllipCRRA, ':', linewidth=4,
            label='ellipse')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'labor supply $n_s$')
    plt.xlim((0, S+1))
    plt.ylim((-2.0, 1.5))
    plt.legend(loc='lower left')
    figname = "images/LC_CompareCRRA"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()


'''
------------------------------------------------------------------------
Solve for optimal elliptical utility of leisure v(l) solution in life
cycle model estimated to fit the CFE specification
------------------------------------------------------------------------
LC_EllipCFE_args      = length 10 tuple, arguments to be passed into
                        solver
start_time_EllipCFE   = scalar > 0, representation of computation start
                        time
LC_EllipCFE_output    = length 10 dictionary, output objects from root
                        finder
elapsed_time_EllipCFE = scalar, elapsed time (in seconds) for
                        computation
n_EllipCFE            = (S,) vector, optimal labor supply over life
                        cycle
b_EllipCFE            = (S-1,) vector, optimal savings over life cycle
b_s_EllipCFE          = (S,) vector, optimal savings/wealth over life
                        cycle for ages s in [1, S] with b1 = 0
b_sp1_EllipCFE        = (S,) vector, optimal savings/wealth over life
                        cycle for ages s in [2, S+1] with b_{S+1} = 0
c_EllipCFE            = (S,) vector, optimal consumption over life cycle
------------------------------------------------------------------------
'''
LC_EllipCFE_args = (S, wbar_CFE, rbar, beta, gamma, e_s,
    chi_EllipCFE, mu_EllipCFE, util_Ellip, cnstr_Ellip)
start_time_EllipCFE = time.clock()
LC_EllipCFE_output = opt.root(LC_EulSolve, guesses,
    args=(LC_EllipCFE_args), method='lm', tol=1e-14)
elapsed_time_EllipCFE = time.clock() - start_time_EllipCFE
n_EllipCFE = LC_EllipCFE_output.x[:S]
b_EllipCFE = LC_EllipCFE_output.x[S:]
b_s_EllipCFE = np.hstack((0, b_EllipCFE))
b_sp1_EllipCFE = np.hstack((b_EllipCFE, 0))
c_EllipCFE = ((1 + rbar) * b_s_EllipCFE +
    wbar_CFE * e_s * n_EllipCFE - b_sp1_EllipCFE)

if plot_LC_EllipCFE:
    # EllipCFE values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_EllipCFE, '-', linewidth=4,
            label='labor supply')
    plt.plot(np.linspace(1, S, S), b_sp1_EllipCFE, '--', linewidth=4,
            label='savings')
    plt.plot(np.linspace(1, S, S), c_EllipCFE, ':', linewidth=4,
            label='consumption')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'optimal values $n_s$, $b_{s+1}$, $c_s$')
    plt.xlim((0, S+1))
    plt.ylim((-4.0, 4.0))
    plt.legend(loc='upper left')
    figname = "images/LC_EllipCFE"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()

    # Ellip, CFE1, and CFE2b labor supply values
    fig, ax = plt.subplots()
    plt.plot(np.linspace(1, S, S), n_CFE1, '-', linewidth=4,
            label='unconstrained')
    plt.plot(np.linspace(1, S, S), n_CFE2b, '--', linewidth=4,
            label='constrained')
    plt.plot(np.linspace(1, S, S), n_EllipCFE, ':', linewidth=4,
            label='ellipse')
    # for the minor ticks, use no labels; default NullFormatter
    minorLocator   = MultipleLocator(1)
    ax.xaxis.set_minor_locator(minorLocator)
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlabel(r'Age $s$')
    plt.ylabel(r'labor supply $n_s$')
    plt.xlim((0, S+1))
    plt.ylim((-0.1, 1.5))
    plt.legend(loc='lower left')
    figname = "images/LC_CompareCFE"
    plt.savefig(figname)
    print("Saved figure: " + figname)
    plt.close()
    # plt.show()
