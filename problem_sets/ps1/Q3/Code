import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


#Lakeshore.txt not found, so I will enter the data manually

temp = [1.4, 4.2, 10, 77, 305]
volt = [1.64, 1.58, 1.38, 1.03, 0.56]
dVdT = [-12.5, -31.6, -26.8, -1.73, -2.3]


#I reversed the lists to have everything in order of increasing V
data = [volt[::-1], temp[::-1], dVdT[::-1]]


def lakeshore(V, data):

    x = np.array(data[0])
    y = np.array(data[1])

    if isinstance(V, (list, tuple)):
    
        spline = interpolate.splrep(x, y)
        temp = interpolate.splev(V, spline)

    else:

        V = [V]
        spline = interpolate.splrep(x, y)
        temp = interpolate.splev(V, spline)
        temp = temp[0]

    #I will use the bootsrap method to find the error on T
    #The "true points" are (x,y), and the interpolated points are(V, temp)
    #The idea is to compare the interpolated values against the true values
    #over many interations

    ran = np.random.default_rng(seed=12345)
    n_resamples = 10
    n_points = 4
    int_points = []
    
    for i in range(n_resamples):

        #Generate a random subset of x to use for resampling. Make sure the subset 
        #in increasing order. Then create a new interpolation for each resample
        
        indices = list(range(len(x)))
        choice = ran.choice(indices, size = n_points, replace = False)
        choice.sort()
        new_int = interpolate.CubicSpline(x[choice], y[choice])
        int_temp = new_int(V)
        int_points.append(int_temp)
        

    #Calculate Error
        
    int_points = np.array(int_points)
    stds = np.std(int_points, axis=0)
    error = np.mean(stds)
    error_std = np.std(stds)
        

    return 'The temperature is {} with an error of {}, std = {}'.format(str(temp), error, error_std)

print(lakeshore([0.56, 1.21, 1.33, 1.64], data))
print(lakeshore(1.03, data))

    

    

