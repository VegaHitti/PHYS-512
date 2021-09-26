import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0.5, 1, 1000)
y = np.log2(x)
tol = 1e-6


#For 0.5 < x < 1, create a truncated Chebyshev polynomial whose error is less than our tolerance

for i in range(150):

    coeffs = np.polynomial.chebyshev.chebfit(x, y, i)
    poly = np.polynomial.chebyshev.chebval(x, coeffs)

    err = [abs(i - j) for i, j in zip(poly, y)]

    mean_err = np.mean(err)

    if mean_err < 1e-6:

        break

    else:

        continue


print("The Chebyshev fit of order {} is accurate within an error 10^-6".format(len(coeffs)-1))


#Now we want to create a function that computes the natural logarithm of any real number.
#There are detailed explanations for this function in the PDF for this question

def mylog2(x):

    mant, exp = np.frexp(x)
    mant_e, exp_e = np.frexp(np.exp(1))
    
    log1 = np.polynomial.chebyshev.chebval(mant, coeffs)
    log2 = exp

    log1_e = np.polynomial.chebyshev.chebval(mant_e, coeffs)
    log2_e = exp_e

    return (log1 + log2)/(log1_e + log2_e)


test_x = np.linspace(1, 10, 100)
test_y = mylog2(test_x)
true_y = np.log(test_x)

plt.plot(test_x, test_y, c="r", label = "Chebyshev")
plt.plot(test_x, true_y, c="b", label = "True Value")

plt.title("Chebyshev Fit of Natural Logarithm vs. True Values")
plt.legend()
plt.show()






