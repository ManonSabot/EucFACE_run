#!/usr/bin/env python
# https://mc3.readthedocs.io/en/latest/mcmc_tutorial.html

import mc3
import numpy as np
import matplotlib.pyplot as plt


def quad(p, x):
    """
    Quadratic polynomial function.

    Parameters
        p: Polynomial constant, linear, and quadratic coefficients.
        x: Array of dependent variables where to evaluate the polynomial.
    Returns
        y: Polinomial evaluated at x:  y(x) = p0 + p1*x + p2*x^2
    """
    print('hello')
    y = p[0] + p[1]*x + p[2]*x**2.0
    return y


print("1")
np.random.seed(314)
x  = np.linspace(0, 10, 1000)
p0 = [3, -2.4, 0.5]
y  = quad(p0, x)

uncert = np.sqrt(np.abs(y))
error  = np.random.normal(0, uncert)
data   = y + error
print("2")
# Define the modeling function as a callable:
func = quad

# List of additional arguments of func (if necessary):
indparams = [x]
print("3")
# Array of initial-guess values of fitting parameters:
params = np.array([ 10.0, -2.0, 0.1])

# Lower and upper boundaries for the MCMC exploration:
pmin = np.array([-10.0, -20.0, -10.0])
pmax = np.array([ 40.0,  20.0,  10.0])
print("4")
# Keep the third parameter fixed:
pstep = np.array([1.0, 0.5, 0.0])

# Make the third parameter share the value of the second parameter:
#It can force a fitting parameter to share its value with another parameter by
#setting its pstep value equal to the negative index of the sharing parameter,
#for example:
#pstep = np.array([1.0, 0.5, -2])

# Parameter prior probability distributions:
# uniform priors
prior    = np.array([ 0.0, 0.0, 0.0])
priorlow = np.array([ 0.0, 0.0, 0.0])
priorup  = np.array([ 0.0, 0.0, 0.0])
print("5")
# Parameter names:
pnames   = ['y0', 'alpha', 'beta']
texnames = [r'$y_{0}$', r'$\alpha$', r'$\beta$']

# Sampler algorithm, choose from: 'snooker', 'demc' or 'mrw'.
sampler = 'snooker'

# MCMC setup:
nsamples =  1e5
burnin   = 1000
nchains  =   14
ncpu     =    7
thinning =    1

# MCMC initial draw, choose from: 'normal' or 'uniform'
kickoff = 'normal'

# DEMC snooker pre-MCMC sample size:
hsize   = 10
print("6")
# Optimization before MCMC, choose from: 'lm' or 'trf':
# Levenberg-Marquardt = lm
leastsq    = 'lm'
chisqscale = False

# MCMC Convergence:
grtest  = True
grbreak = 1.01
grnmin  = 0.5

# Carter & Winn (2009) Wavelet-likelihood method:
wlike = False

fgamma   = 1.0  # Scale factor for DEMC's gamma jump.
fepsilon = 0.0  # Jump scale factor for DEMC's "e" distribution

# Logging:
log = 'MCMC_tutorial.log'
print("7")
# File outputs:
savefile = 'MCMC_tutorial.npz'
plots    = True
rms      = True
print("8")
# Run the MCMC:
mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
                        indparams=indparams, pmin=pmin, pmax=pmax, pstep=pstep,
                        pnames=pnames, texnames=texnames,
                        prior=prior, priorlow=priorlow, priorup=priorup,
                        sampler=sampler, nsamples=nsamples,  nchains=nchains,
                        ncpu=ncpu, burnin=burnin, thinning=thinning,
                        leastsq=leastsq, chisqscale=chisqscale,
                        grtest=grtest, grbreak=grbreak, grnmin=grnmin,
                        hsize=hsize, kickoff=kickoff,
                        wlike=wlike, log=log,
                        plots=plots, savefile=savefile, rms=rms)
print("9")
