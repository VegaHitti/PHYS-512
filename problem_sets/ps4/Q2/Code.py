import numpy as np
import camb
from matplotlib import pyplot as plt
import time
import scipy.stats as scs


#I used the formula for derivatives that was derived in ps1 question 1

def deriv(f, pars, dx):

    #take derivative wrt each parameter

    grad = np.zeros([2507, len(pars)]) #2507 is just the length of the "x" values

    for i in range(len(pars)):

        pars_plus = pars.copy()
        pars_plus[i] += dx

        pars_minus = pars.copy()
        pars_minus[i] -= dx

        pars_2plus = pars.copy()
        pars_2plus[i] += 2*dx

        pars_2minus = pars.copy()
        pars_2minus[i] -= 2*dx

        dfdx = ((2/3)*(f(pars_plus) - f(pars_minus)) + (1/12)*(f(pars_2minus) - f(pars_2plus)))/dx

        dfdx = dfdx[:2507]

        grad[:,i] = dfdx

    return grad


def fit_newton(f, pars, y, dx, n_iter = 10):

    #Newton's method, as shown in class

    for i in range(n_iter):

        model = f(pars)
        derivs = deriv(f, pars, dx)
        res = y - model
        lhs = derivs.T@derivs
        rhs = derivs.T@res
        dpars = np.linalg.pinv(lhs)@rhs
        pars = pars + dpars
        chisq = np.sum(res**2)
        print("Chi Square value is {}".format(chisq))

    #Not sure if this is the correct way to calculate the error on each parameter
    #At first, I took the square root of the diagonal of the curvature matrix,
    #but this yielded ridiculous errors to the order of 10^8 or more.
    #So, I resorted to this method (similar to ps3 question 3, which involved least squares fitting)
    
    noise = np.std(res)
    N = np.eye(2507)*noise**2
    N_inv = np.linalg.inv(N)
    mat = derivs.T@N_inv@derivs
    err_mat = np.linalg.inv(mat)
    errs = np.sqrt(np.diag(err_mat))

    return pars, errs


#The function below is from Sievers' starter code (from question 1)

def get_spectrum(pars, lmax=3000):
    
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    new_pars=camb.CAMBparams()
    
    new_pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    new_pars.InitPower.set_params(As=As,ns=ns,r=0)
    new_pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    
    results=camb.get_results(new_pars)
    powers=results.get_cmb_power_spectra(new_pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    #you could return the full power spectrum here if you wanted to do say EE
    
    return tt[2:2509]


#I used the second set of pars from question 1 as an initial guess
pars=np.asarray([69,0.022,0.12,0.06,2.1e-9, 0.95])

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)

ell=planck[:,0]

spec=planck[:,1]

errs=0.5*(planck[:,2]+planck[:,3]);

dx = 1e-12

fit_pars, fit_errs = fit_newton(get_spectrum, pars, spec, dx, 19)

model = get_spectrum(fit_pars)

model = model[:len(spec)]

resid = spec-model

chisq = np.sum((resid/errs)**2)

dof = len(resid) - len(fit_pars)

pval = scs.chi2.sf(chisq, dof)


print("The best-fit parameters are {}, with errors {}, and the p-value is {}".format(fit_pars, fit_errs, pval))

with open("planck_fit_params.txt", "w") as f:

    f.write("The best-fit parameters are {}, with errors {}, and the p-value is {}".format(fit_pars, fit_errs, pval))
