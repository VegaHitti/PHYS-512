import numpy as np

eps = 2**-52

def optimal_dx(fun, x):

    dx = 0.01 #choose arbitrary dx to initiate process

    
    #repeat process 3 times for a better estimate
    for i in range(3):

        #Equation derived in written document "phys512-q2.pdf"
        f3 = (fun(x+2*dx) + 3*fun(x) - 3*fun(x+dx) - fun(x-dx))/(dx**3)

        dx = np.abs(eps*fun(x)/f3)**(1/3)

    return dx


def deriv(fun, x, dx):

    return (fun(x+dx)-fun(x-dx))/(2*dx)
    

def ndiff(fun, x, full = False):

    dx = optimal_dx(fun, x)

    d = deriv(fun, x, dx)

    f3 = (fun(x+2*dx) + 3*fun(x) - 3*fun(x+dx) - fun(x-dx))/(dx**3)

    #Truncation and roundoff error
    err = np.abs(f3*dx**2 + eps*fun(x)/dx)

    #Error in terms of percentage
    err_f = err/d*100

    if full:

        return 'The derivative is {},  dx is {}, and the error is {} or {}%'.format(d,dx,err,err_f)
    
    else:
        
        return 'The derivative is {}'.format(d)


#Example to demonstrate that it works    
fun = np.exp    
x = np.linspace(1,5,5)
ans = ndiff(fun, x, full=True)

print(ans)





    
   
