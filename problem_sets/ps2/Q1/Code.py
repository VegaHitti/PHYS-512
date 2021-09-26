import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

#The following is very similar to some code that was shown in class

#Make a Legendre Polynomial matrix that is n x n

def legendre(n):

    x = np.linspace(-1, 1, n)
    mat = np.zeros([n ,n])
    mat[:,0] = 1.0
    mat[:,1] = x

    if n > 2:

        for i in range(1, n - 1):

            mat[:,i+1] = ((2.0*i+1)*x*mat[:,i]-i*mat[:,i-1])/(i+1.0)

    return mat
    

#Find integration coefficients using said matrix

def coefficients(n):

    mat = legendre(n)
    mat_inv = np.linalg.inv(mat)
    coeffs = mat_inv[0,:]
    coeffs = coeffs/coeffs.sum()*(n-1.0)

    return coeffs
    

#Use the coefficients to integrate

def integrate(fun, xmin, xmax, d, order=2):

    coeffs = coefficients(order + 1)
    npts = int((xmax-xmin)/d) + 1
    m = (npts-1)%order

    if m > 0:

        npts = npts + (order - m)

    assert(npts%(order)==1)

    npts = int(npts)
    
    x = np.linspace(xmin, xmax, npts)
    dx = np.median(np.diff(x))
    data = fun(x)
    
    
    #We reshape the data, and then sum over the columns. But first, we add the
    #last point to the first point and double the first column. We add because
    #the endpoints only show up once, and we double because each element
    #appears as the last element in the previous row.

    mat = np.reshape(data[:-1], [(npts-1)//order, order]). copy()
    mat[0,0] = mat[0,0] + data[-1]
    mat[1:,0] = 2*mat[1:,0]

    vec = np.sum(mat, axis = 0)
    tot = np.sum(vec*coeffs[:-1])*dx

    return tot



R = 0.15
e_o = 8.854e-12
sig = 2e-11

z = np.linspace(0, 0.5, 15)
z = np.append(z, R)
z.sort()

dx = 0.1
t_min = 0
t_max = np.pi

homemade = []
premade = []
true_vals = []


#The function below is the function to integrate. It is the electric field cause by a charged 
#spherical shell at various values of z. The derivation of this function is found in the PDF document
#for this question

for i in z:

    fun = lambda theta: sig*R**2/(2*e_o)*(i-R*np.cos(theta))*np.sin(theta)/(R**2 + i**2 - 2*R*i*np.cos(theta))**(3/2)

    val1 = integrate(fun, t_min, t_max, dx, 10)
    val2 = quad(fun, t_min, t_max)
    val3 = R**2*sig/(2*e_o*i**2)*((i-R)/abs(i-R) - (-i-R)/abs(i+R))

    homemade.append(val1)
    premade.append(val2)
    true_vals.append(val3)


premade2 = []

for group in premade:

    premade2.append(group[0])
    
    
#Now I test my integrator vs scipy's integrator and the true values
#The bounds of the integration are zero to pi

fig, ax = plt.subplots(2, 2, figsize=(10,16))


ax[0,0].plot(z, homemade, "o", markersize=3, c="b")
ax[0,0].set_title("Integrator Values")

ax[0,1].plot(z, premade2, "o",markersize=3, c="r")
ax[0,1].set_title("Scipy Values")

ax[1,0].plot(z, true_vals, "o", markersize=3, c ="k")
ax[1,0].set_title("True Values")

ax[1,1].plot(z, homemade, "o", markersize=3, c="b", label = "Integrator")
ax[1,1].plot(z, premade2, "o", markersize=3, c="r", label = "Scipy")
ax[1,1].plot(z, true_vals, "o", markersize=3, c="k", label = "True Values" )
ax[1,1].set_title("Combined Plots")
ax[1,1].legend(prop = {"size":7})

plt.show()



        

        

        
    
        
    
