import numpy as np

#First, verify for f(x) = exp(x)

#x = 42 is easily verifiable since it was used in class

x = 42
eps = 2**(-52)
dx = eps**(1/3) 
#dx = (eps*f/f''')**(1/3) = (eps*e**x/e**x)**(1/3)= eps**(1/3)

x1 = x + dx
x2 = x - dx
x3 = x + 2*dx
x4 = x - 2*dx

f1 = np.exp(x1)
f2 = np.exp(x2)
f3 = np.exp(x3)
f4 = np.exp(x4)

deriv1 = (f1 + f3 - f2 - f4)/(6*dx)

print("Derivative is ",deriv1," with fractional error ",1-deriv1/np.exp(x))

#Now we verify for f(x) = exp(0.01x)

dx2 = eps^(1/3)/0.01 
#dx = (eps*f/f''')**(1/3) = (eps*exp(0.01x)/(0.01**3*exp(0.01x))**(1/3) = eps**(1/3)/0.01

f5 = np.exp(0.01*x1)
f6 = np.exp(0.01*x2)
f7 = np.exp(0.01*x3)
f8 = np.exp(0.01*x4)

deriv2 = (f5 + f7 - f6 - f8)/(6*dx2)

print("Derivative is ",deriv2," with fractional error ",1-deriv2/np.exp(0.01*x))


