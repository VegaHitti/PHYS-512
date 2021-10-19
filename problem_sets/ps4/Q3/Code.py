import matplotlib.pyplot as plt
import numpy as np
import camb


def get_spectrum(pars,lmax=3000):
    
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    
    return tt[2:]


planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)

ell=planck[:,0]

spec=planck[:,1]

errs=0.5*(planck[:,2]+planck[:,3]);

#The fit parameters from question 2

fit_pars = np.asarray([6.84955561e+01,2.24189493e-02,1.16905708e-01,7.55198229e-03,1.89729855e-09,9.72141191e-01])

#The curvature matrix from question 2

curv_mat = np.asarray([[ 1.94342347e-04,3.07301700e-08,-1.10572151e-07,-1.42989789e-08,5.65956458e-17,4.19235357e-07],[3.07301700e-08,7.05911894e-11,3.27729894e-11,3.12091854e-12,-9.28128673e-21,1.83090184e-10],[-1.10572151e-07,3.27729894e-11,6.89153898e-10,-3.41367050e-10,1.29566297e-18,-3.21372256e-10],[-1.42989789e-08,3.12091854e-12,-3.41367050e-10,3.63776065e-09,5.67577362e-18,2.09822248e-10],[5.65956458e-17,-9.28128673e-21,1.29566297e-18,5.67577362e-18,5.23502642e-26,-7.42239946e-19],[ 4.19235357e-07,1.83090184e-10,-3.21372256e-10,2.09822248e-10,-7.42239946e-19,2.29292073e-08]])

par_errs = np.sqrt(np.diag(curv_mat))

                   
def probability(pars, y, y_errs):

    model = get_spectrum(pars)
    probs = []

    #The probability is a Gaussian, but we'll consider log(P) (it works out better)
    for i in range(len(y)):

        prob = np.log(1/np.sqrt(2*np.pi*y_errs[i])) * (-(y[i]-model[i])**2/(2*y_errs[i]**2))
        probs.append(prob)

    return np.sum(probs)


#This function assigns a probability distribuition function to each parameter
#The MCMC chain will need upper/lower limits on the parameters
#Let's set these limits to +/- 3 sigma

def prior_pdf(pars, fit_pars = fit_pars, fit_errs = par_errs):

    u_lim = fit_pars + 3*fit_errs
    l_lim = fit_pars - 3*fit_errs

    w = 0

    for i, cons in enumerate(pars):

        if l_lim[i] < cons < u_lim[i]:

            continue

        else:

            w += 1

    if w ==0:

        return 1

    return 0


def next(pars, y, y_errs):

    prior = prior_pdf(pars)

    if prior == 0:

        return 0

    return prior*probability(pars, y, y_errs)


#Metropolis-Hastings method of MCMC

def mcmc(pars_list, n_iter = 20000):

    for i in range(n_iter-1):

        pars_cur = pars_list[-1]

        trial_step = np.zeros_like(pars_cur)

        #Create a trial step
        
        for i in range(pars_cur.size):

            trial_step[i] = np.random.normal(loc = pars_cur[i], scale = par_errs[i], size = 1)

        prob1 = next(pars_cur, spec, errs)
        prob2 = next(trial_step, spec, errs)
        ratio = prob2/prob1

        #Choose a random number to then decide if the step is accepted 
        
        n = np.random.uniform()

        if ratio > n:

            pars_list.append(trial_step)

        else:

            pars_list.append(pars_cur)
            
        with open ("planck_chain.txt", "a") as f:

            for j in pars_list[-1]:

                f.write(f"{j}\t")

            f.write("\n")


    return pars_list


pars_list = [fit_pars]

n_iter = 20000

chain = mcmc(pars_list, n_iter)

best_pars = np.mean(chain) #This value is reported in the PDF file
err_best_pars = np.std(chain) #This value is reported in the PDF file






    

                   
