import numpy as np

#The code below is very similar to some code that was shown in class

def integrate_adaptive(fun, x0, x1, tol, extra=None):

    x = np.linspace(x0, x1, 5)
    dx = (x1-x0)/(len(x)-1)

    if extra == None:
    
        y = fun(x)

    else:

        #See the explanation of this step in the PDF 
        
        y = []
        y.append(extra[0])
        y.append(extra[1])
        y.append(extra[2])
        y.append(fun(x[1]))
        y.append(fun(x[3]))
        y.sort()
        

    #Computation of Simpson's rule
    
    area1 = 2*dx/3*(y[0] + 4*y[2] + y[4])
    area2 = dx/3*(y[0] + 4*y[1] + 2*y[2] + 4*y[3] + y[4])

    err = abs(area1-area2)

    if err < tol:
        
        
        return area2

    else:
        
        #Now we split the interval in half and start again on each side.
        #See the PDF for an explanation of the last paramater (the list of y's).
        
        mid = (x0+x1)/2
        left = integrate_adaptive(fun, x0, mid, tol/2, [y[0], y[1], y[2]])
        right = integrate_adaptive(fun, mid, x1, tol/2, [y[2], y[3], y[4]])

        return left + right
    

#Testing a random function

def func(x):

    return 3*x**4 + 2*x

x0 = 1
x1 = 5
tol = 1e-7

a = integrate_adaptive(func, x0, x1, tol)

print("The integral is: {}".format(a))
    

    
    

    
    
