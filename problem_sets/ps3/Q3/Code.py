import numpy as np
import matplotlib.pyplot as plt

#Part b)

def fun(x, y, pars):

    z = pars[0]*(x**2 + y**2) + pars[1]*x + pars[2]*y + pars[3]

    return z

data = np.loadtxt("dish_zenith.txt", unpack = True)

x = data[0]
y = data[1]
z = data[2]

nd = len(x)
nm = 4

A = np.zeros([nd, nm])
A[:,0] = x**2 + y**2
A[:,1] = x
A[:,2] = y
A[:,3] = 1

#Using m= [A^T*N^(-1)*A]^(-1)*A^T*N^(-1)*z

lhs = A.T@A
rhs = A.T@z
fit_pars = np.linalg.inv(lhs)@rhs
z_fit = A@fit_pars

fig = plt.figure(figsize = (5, 8))
ax = plt.axes(projection='3d')

ax.scatter3D(x, y, z, c="k")
ax.plot_trisurf(x, y, z_fit, cmap="rainbow")

plt.show()
fig.savefig("paraboloid.png")

print("Parameters [a,b,c,d] are: {}".format(fit_pars))


#Part c)

#Finding the noise


residuals = z - z_fit
noise = np.std(residuals)

print("The noise is {}".format(noise))


#Finding the error in parameters
#As derived in class, the error is given by (A^T*N^(-1)*A)^(-1)


N = np.eye(len(x))*noise**2
N_inv = np.linalg.inv(N)
mat = A.T@N_inv@A
err_mat = np.linalg.inv(mat)
errs = np.sqrt(np.diag(err_mat))


print("The errors on [a,b,c,d] are: {}".format(errs))


#Finding the focal length


x0 = -fit_pars[1]/(2*fit_pars[0])
y0 = -fit_pars[2]/(2*fit_pars[0])
z0 = fit_pars[3] - (fit_pars[1])**2/(4*fit_pars[0]) - (fit_pars[2])**2/(4*fit_pars[0])

f = ((x-x0)**2 + (y-y0)**2)/(4*(z_fit - z0))

f_avg = np.mean(f)
std_f = np.std(f)

print("The focal length is {} with std {}".format(f_avg, std_f))


#Finding the error bars on the focal point


z_fit_errs = A@err_mat@A.T
z_fit_sig = np.sqrt(np.diag(z_fit_errs)) #this is my error in z

x0_min = -(fit_pars[1]+errs[1])/(2*(fit_pars[0]-errs[0]))
x0_max = -(fit_pars[1]-errs[1])/(2*(fit_pars[0]+errs[0]))

y0_min = -(fit_pars[2]+errs[2])/(2*(fit_pars[0]+errs[0]))
y0_max = -(fit_pars[2]-errs[2])/(2*(fit_pars[0]-errs[0]))


z0_min = (fit_pars[3]-errs[3]) - (fit_pars[1]+errs[1])**2/(4*(fit_pars[0]-errs[0])) - (fit_pars[2]+errs[2])**2/(4*(fit_pars[0]+errs[0]))
z0_max = (fit_pars[3]+errs[3]) - (fit_pars[1]-errs[1])**2/(4*(fit_pars[0]+errs[0])) - (fit_pars[2]-errs[2])**2/(4*(fit_pars[0]-errs[0]))

ex = np.abs((x0_min -  x0_max)/2) #error in x0
ey = np.abs((y0_min -  y0_max)/2) #error in y0
ez = np.abs((z0_min -  z0_max)/2) #error in z0


def f(x, y, z, x_0, y_0, z_0):

    return ((x-x0)**2 + (y-y0)**2)/(4*(z_fit - z0))


#Since x, y, z can be positive or negative, it is harder to tell which parameter will give f_min, f_max.
#So, I've listed all the possibilities, and I will determine f_min, f_max from here.

fs = [f(x, y, z, x0+ex, y0+ey, z0+ez), f(x, y, z, x-ex, y+ey, z+ez), f(x, y, z, x+ex, y-ey, z+ez), f(x, y, z, x+ex, y+ey, z-ez),f(x,y,z, x+ex, y-ey, z-ez), f(x,y,z, x-ex, y-ey, z+ez), f(x,y,z, x-ex, y+ey, z-ez), f(x,y,z, x-ex, y-ey, z-ez)]  

fs_avg = [np.mean(i) for i in fs]

f_min = min(fs_avg)
f_max = max(fs_avg)
err_f = np.abs(f_min - f_max)/2

print("The min focal length is {}, and the max focal length is {}".format(f_min, f_max))
print("f = {} plus or minus {}".format(f_avg, err_f))






