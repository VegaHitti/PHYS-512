import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

#Part a)

#Below, I've converted the half-lives of all the products in the U238 chain into seconds

hl = [1.409e17, 2.082e6, 24100, 7.742e12, 2.377e12, 5.046e10, 330350, 186, 1608,1194, 1.643e-4, 7.033e8, 1.5815e8, 1.1956e7] 

def fun(t, y, half_life = hl):

    dydt = np.zeros(len(half_life) + 1)

    dydt[0] = -y[0]/half_life[0]
    dydt[-1] = y[-2]/half_life[-1] 

    for i in range(len(dydt)-2):

        dydt[i+1] = y[i]/half_life[i] - y[i+1]/half_life[i+1]

    return dydt


n = 15
y0 = np.zeros(n)
y0[0] = 1
t0 = 0
t1 = 8e17
t = np.linspace(t0, t1, 1000000)

sol = integrate.solve_ivp(fun, [t0, t1], y0, method = "Radau", t_eval = t)


#Part b)


#U238 is the first species in the chain, and Pb206 is the last

plt.plot(sol.t, sol.y[0,:], c="lightseagreen", label = "U238")
plt.plot(sol.t, sol.y[-1,:], c="mediumslateblue", label = "Pb206")
plt.title("Ratio of Pb206 to U328 Over Time")
plt.legend()
plt.show()
plt.savefig("ratio1.png")


#U234 is the 4th species in the chain, and Thorium 230 is the 5th

plt.plot(sol.t, sol.y[3,:], c = "lightseagreen", label = "U234")
plt.plot(sol.t, sol.y[4,:], c = "mediumslateblue", label = "Thorium 230")
plt.title("Ratio of Thorium 230 to U234 Over Time")
plt.legend()
plt.show()
plt.savefig("ratio2.png")







    
