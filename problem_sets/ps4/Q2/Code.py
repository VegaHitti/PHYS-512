import numpy as np
import camb
from matplotlib import pyplot as plt
import time
import scipy.stats as scs


#I used the formula for derivatives that was derived in ps1 question 1

def deriv(f, pars, dx):

    #Take derivative wrt each parameter

    grad = np.zeros([2507, len(pars)]) #2507 is just the length of the "x" values

    for i in range(len(pars)):

        pars_plus = pars.copy()
        pars_plus[i] += dx[i]

        pars_minus = pars.copy()
        pars_minus[i] -= dx[i]

        pars_2plus = pars.copy()
        pars_2plus[i] += 2*dx[i]

        pars_2minus = pars.copy()
        pars_2minus[i] -= 2*dx[i]

        dfdx = ((2/3)*(f(pars_plus) - f(pars_minus)) + (1/12)*(f(pars_2minus) - f(pars_2plus)))/dx[i]

        dfdx = dfdx[:2507]

        grad[:,i] = dfdx

    return grad


#The function below was written in class
def update_lamda(lamda,success):
    
    if success:
        
        lamda=lamda/1.5
        
        if lamda<0.5:
            
            lamda=0
            
    else:
        
        if lamda==0:
            
            lamda=1
            
        else:
            
            lamda=lamda*1.5**2
            
    return lamda


#The function below is similar to one seen in class
#I tried running the code without considering noise,
#but the results were awful.
def fit_lm(f, pars, y, dx, errs, n_iter = 10):

    lamda = 0
    model = f(pars)[:2507]
    chisq_old = np.sum(((model-y)/errs)**2)

    #Noise matrix
    N = np.zeros((2507, 2507))
    for i in range(len(N)):
        N[i,i] = errs[i]

    N_inv = np.linalg.inv(N)

    for i in range(n_iter):

        model = f(pars)[:2507]
        derivs = deriv(f, pars, dx)
        res = y - model
        lhs = derivs.T@N_inv@derivs
        lhs = lhs + lamda*np.diag(np.diag(lhs))
        rhs = derivs.T@N_inv@res
        curv_mat = np.linalg.inv(lhs) 
        dpars = curv_mat@rhs
        pars_trial = pars + dpars

        model = f(pars_trial)
        chisq = np.sum(((y-model)/errs)**2)

        #Choose if we accept step
        if (chisq < chisq_old):

            lamda = update_lamda(lamda, True)

            pars = pars + dpars

        else:

            lamda = update_lamda(lamda, False)

        print("On iteration {}, chisq is {}".format(i, chisq))

    return pars, curv_mat, chisq


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

#Scale dx wrt to each parameter
dx = [1e-3, 1e-7, 1e-6, 1e-7, 1e-14, 1e-6]

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)

ell=planck[:,0]

spec=planck[:,1]

errs=0.5*(planck[:,2]+planck[:,3]);

fit_pars, curv_mat, fit_chisq = fit_lm(get_spectrum, pars, spec, dx, errs, n_iter = 9)

fit_errs = np.sqrt(np.diag(curv_mat))

fit_model = get_spectrum(fit_pars)

res = spec - fit_model

dof = len(res) - len(fit_pars)

pval = scs.chi2.sf(fit_chisq, dof)

with open("planck_fit_params.txt", "w") as f:

    f.write("The best-fit parameters are {}, with errors {}.\n Chisq is {}, and the p-value is {}.\n The curvature matrix is {}.".format(fit_pars, fit_errs, fit_chisq, pval, curv_mat))
    

fig = plt.figure()

plt.plot(ell, spec, "o",markersize = 1.5, c="k", label = "Data")
plt.plot(ell, fit_model, c="skyblue", label = "LM Fit")
plt.title("Levenberg-Marquardt Fit Vs. Data")
plt.legend()
plt.show()

fig.savefig("q2.png")


