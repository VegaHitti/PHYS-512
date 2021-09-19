import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#First I will fit cos(x) with all three methods


# 1) Polynomial fit

#The code here is very similar to some code that was presented in class

n_pts = 8
x = np.linspace(-np.pi/2, np.pi/2, n_pts)
y = np.cos(x)


#Find the coefficient matrix c

xx = np.empty([n_pts, n_pts])

for i in range(n_pts):

    xx[:,i] = x**i

xx_inv = np.linalg.inv(xx)

c = xx_inv @ y

y_pred = xx @ c

err = np.std(y_pred-y)


#Creating a fitted curve with the matrix found above

new_x = np.linspace(-np.pi/2, np.pi/2, 2001)

new_xx = np.empty([len(new_x), n_pts])

for i in range(n_pts):

    new_xx[:,i] = new_x**i

poly_y = new_xx @ c

true_y = np.cos(new_x)


# 2) Cubic Spline

#I will use splrep and splev from scipy.interpolate
#For this, I can reuse x = np.linspace(-np.pi/2, np.pi/2, n_pts) and  y = np.cos(x)

xxx = np.linspace(x[0], x[-1], 2001)

spline = interpolate.splrep(x, y)
spline_y = interpolate.splev(xxx, spline)
true_y2 = np.cos(xxx)


# 3) Rational fit

#Again, the code below is very similar to some code that was shown in class


def rat_eval(p, q, x):
    
    top=0
    
    for i in range(len(p)):
        
        top=top+p[i]*x**i
        
    bot=1
    
    for i in range(len(q)):
        
        bot=bot+q[i]*x**(i+1)
        
    return top/bot


def rat_fit(x,y,n,m):
    
    assert(len(x)==n+m-1)
    
    assert(len(y)==len(x))
    
    mat=np.zeros([n+m-1,n+m-1])
    
    for i in range(n):
        
        mat[:,i]=x**i
        
    for i in range(1,m):
        
        mat[:,i-1+n]=-y*x**i
        
    pars=np.dot(np.linalg.inv(mat),y)
    
    p=pars[:n]
    
    q=pars[n:]
    
    return p, q


#x = np.linspace(-np.pi/2, np.pi/2, n+m-1) for the rational fit
#And since n_pts = 9 for the other fits, we want n + m - 1 = 9
#So again, I can use x and y from above to call rat_fit()

n = 4
m = 5

p, q = rat_fit(x, y, n, m)

#I will use xxx = np.linspace(x[0], x[-1], 2001) from the
#Cubic Spline fit to call rat_eval()

rat_y = rat_eval(p, q, xxx)


#I will make a rough estimation of the error of each fit to then compare
#them. To do this, I will take the difference between the real y values
#and the interpolated ones at several points. Then, I will take the mean
#of these differences.

n_samples = 100
ran = np.random.default_rng(seed=12345)
indices = list(range(len(xxx)))
choice = ran.choice(indices, size = n_samples, replace = False)


errs_poly = [poly_y[i]-true_y2[i] for i in choice]
errs_spline = [spline_y[i]-true_y2[i] for i in choice]
errs_rat = [rat_y[i]-true_y2[i] for i in choice]

err_poly = np.mean(errs_poly)
err_spline = np.mean(errs_spline)
err_rat = np.mean(errs_rat)

print("Polynomial error = {}, Spline error = {} and Rational error = {}".format(err_poly, err_spline, err_rat))



#Plot all functions to visualize if they are accurate fits

fig, ax = plt.subplots(1, 3)

ax[0].plot(new_x, poly_y, c="b", label="Polynomial fit")
ax[0].plot(xxx, true_y2, c = "r", label="Function")
ax[0].plot(x, y, "o", label="Data points")

ax[0].legend()

ax[1].plot(xxx, spline_y, c="b", label="Cubic Spline")
ax[1].plot(xxx, true_y2 , c = "r", label="Function")
ax[1].plot(x, y, "o", label="Data points")

ax[1].legend()

ax[2].plot(xxx, rat_y, c="b", label="Rational fit")
ax[2].plot(xxx, true_y2 , c = "r", label="Function")
ax[2].plot(x, y, "o", label="Data points")

ax[2].legend()

plt.show()
