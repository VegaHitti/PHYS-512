import numpy as np
import matplotlib.pyplot as plt

#There isn't much to comment about the code below. Derivations are in the PDF file.


def f(x,y):

    return y/(1+x**2)


def rk4_step(fun, x, y, h):

    k1 = h*fun(x, y)
    k2 = h*fun(x+h/2, y+k1/2)
    k3 = h*fun(x+h/2, y+k2/2)
    k4 = h*fun(x+h/2, y+k3)

    dy = (k1 + 2*k2 + 2*k3 + k4)/6

    return y + dy


n_pts = 20001

x = np.linspace(-20, 20, n_pts)

y = np.zeros(n_pts)

y[0] = 1 #y(-20) = 1

h = np.abs((x[-1] - x[0])/n_pts)


for i in range(n_pts - 1):

    y[i+1] = rk4_step(f, x[i], y[i], h)
    

true_val = np.exp(np.arctan(20))*np.exp(np.arctan(x)) #As derived in PDF

print(np.std(true_val - y)) #Check if RK4 is a good estimate


fig, ax = plt.subplots(2, 2, figsize=(10,16))

ax[0,0].plot(x, y, c="b")
ax[0,0].set_title("RK4 Estimate")

ax[0,1].plot(x, true_val, c="r")
ax[0,1].set_title("True Value")

ax[1,0].plot(x, y, c="b", label = "RK4 Estimate")
ax[1,0].plot(x, true_val, c="r", label = "True Value")
ax[1,0].set_title("RK4 Estimate vs. True Value")
ax[1,0].legend(prop={"size":9})

ax[1,1].plot(x, y-true_val, c="k")
ax[1,1].set_title("RK4 Error")

plt.show()
fig.savefig("rk4.png")
