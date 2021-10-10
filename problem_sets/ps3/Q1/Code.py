import numpy as np
import matplotlib.pyplot as plt

#There isn't much to comment about the code below. Derivations are in the PDF file.


def f(x,y):

    return y/(1+x**2)


def rk4_step(fun, x, y, h):

    k1 = h*fun(x, y)
    k2 = h*fun(x+h/2, y+k1/2)
    k3 = h*fun(x+h/2, y+k2/2)
    k4 = h*fun(x+h, y+k3)

    dy = (k1 + 2*k2 + 2*k3 + k4)/6

    return y + dy


n_pts = 201

x = np.linspace(-20, 20, n_pts)

y = np.zeros(n_pts)

y[0] = 1 #y(-20) = 1

h = np.abs((x[-1] - x[0])/n_pts)

for i in range(n_pts - 1):

    y[i+1] = rk4_step(f, x[i], y[i], h)

true_val = np.exp(np.arctan(20))*np.exp(np.arctan(x)) #As derived in PDF


print("The std for the RK4 is {}".format(np.std(true_val - y)))

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


#Part 2 begins here

#I know that rk4_stepd() is not written very efficiently,
#but for some reason I couldn't make it work by calling rk4_step()

def rk4_stepd(fun, x, y, h):

    k1y1 = h*fun(x, y)
    k2y1 = h*fun(x+h/2, y+k1y1/2)
    k3y1 = h*fun(x+h/2, y+k2y1/2)
    k4y1 = h*fun(x+h, y+k3y1)

    dy1 = (k1y1 + 2*k2y1 + 2*k3y1 + k4y1)/6

    k1y2a = h/2*fun(x, y)
    k2y2a = h/2*fun(x+h/4, y+k1y2a/2)
    k3y2a = h/2*fun(x+h/4, y+k2y2a/2)
    k4y2a = h/2*fun(x+h/2, y+k3y2a)

    dy2a = (k1y2a + 2*k2y2a + 2*k3y2a + k4y2a)/6

    k1y2b = h/2*fun(x+h/2, y+dy2a)
    k2y2b = h/2*fun(x+h/4, y+dy2a+k1y2b/2)
    k3y2b = h/2*fun(x+h/4, y+dy2a+k2y2b/2)
    k4y2b = h/2*fun(x+h, y+dy2a+k3y2b)

    dy2b = (k1y2b + 2*k2y2b + 2*k3y2b + k4y2b)/6

    ans = y + (-1/15)*dy1 + (16/15)*(dy2a+dy2b) #This linear combination is derived in the PDF 

    return ans


n_d = 68

x_d = np.linspace(-20, 20, n_d)

y_d = np.zeros(n_d)

y_d[0] = 1

h_d = np.abs((x_d[-1] - x_d[0])/n_d)

for i in range(n_d - 1):

    y_d[i+1] = rk4_stepd(f, x_d[i], y_d[i], h_d)


true_val_d = np.exp(np.arctan(20))*np.exp(np.arctan(x_d))
    

print("The std for the half-step RK4 is {}".format(np.std(true_val_d - y_d)))


fig2, ax2 = plt.subplots(2, 2, figsize=(10,16))


ax2[0,0].plot(x_d, y_d, c="b")
ax2[0,0].set_title("Half-step RK4 Estimate")

ax2[0,1].plot(x_d, true_val_d, c="r")
ax2[0,1].set_title("True Value")

ax2[1,0].plot(x_d, y_d, c="b", label = "RK4 Estimate")
ax2[1,0].plot(x_d, true_val_d, c="r", label = "True Value")
ax2[1,0].set_title("Half-step RK4 Estimate vs. True Value")
ax2[1,0].legend(prop={"size":9})

ax2[1,1].plot(x_d, y_d-true_val_d, c="k")
ax2[1,1].set_title("Half-step RK4 Error")

plt.show()

fig2.savefig("rk4_d.png")









