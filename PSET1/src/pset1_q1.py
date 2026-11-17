######################################
#### MODULES AND PATHS ####
######################################

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize 
from scipy import stats
import statsmodels.api as sm 
from pathlib import Path
from math import pi
from math import log
import random


MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"

for directory in (DATA, CODE):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(20)

######################################
#### Question 1 ####
######################################

###################
#### 1.a ####
###################

# read in dataset for q1
df1 = pd.read_csv(DATA / "data1.dat", sep = "\s\s", header = None)

# We'll calculate the OLS coefficient and SE using the well-known formulas manually and then also using the built-in package. 

df1[["cons"]] = 1

# Let's set the columns names
df1.columns = ["y", "x", "cons"]

Y1 = df1[["y"]].to_numpy()
X1 = df1[["cons", "x"]].to_numpy()

# Let's first get the coefficient estimates
ols_coefs_manual = np.linalg.inv(X1.T @ X1) @ X1.T @ Y1
print(f"The coefficient esimates for theta_1 is {ols_coefs_manual[0]} and for theta_2 is {ols_coefs_manual[1]}")


# Now, we will get the standard error estimates. First, we need to recover sigma^2 hat, which in turn is based on our estimated residuals
e_hat = Y1 - (X1 @ ols_coefs_manual)

# Now we get sigma^2 hat
sigma2_hat = (1 / len(df1)) * e_hat.T @ e_hat

# we know that var(beta | x) = sigma2 hat (X'X)^-1
cov_var_ols = np.linalg.inv(X1.T @ X1)

# we need the diagonal elements
cov_var_ols = np.linalg.diagonal(cov_var_ols)

# Finally, we get our SE
se_ols = np.sqrt(sigma2_hat * vars)
se_ols = np.reshape(se_ols, 2)

print(f"The SE for theta_1 is {se_ols[0]} and for theta_2 is {se_ols[1]}")

###################
#### 1.b ####
###################

# We'll define our log likelihood. We can inverse the linear model to have that y_i ~ N(theta_1 + theta_2 * x_i, sigma^2). Note we'll use the negative log-likelihood given that we are minimizing 

# Although we can analytically compute the derivative, for practice we will use the one-sided numeric approximation Dan describes in the pset

def normal_e_nll(params, *args): 
    
    n = len(df1)
    theta_1 = params[0]
    theta_2 = params[1]
    sigma = params[2]

    y = args[0]
    x = args[1]

    nll = (n / 2) * log(2 * pi) + n * log(sigma) + (1 / (2 * sigma ** 2)) * sum((y - theta_1 - theta_2 * x) ** 2)
    
    return nll

def normal_e_nll_i(params, *args): 

    n = len(df1)
    theta_1 = params[0]
    theta_2 = params[1]
    sigma = params[2]

    y = args[0]
    x = args[1]

    nll_i = (1 / 2) * log(2 * pi) + log(sigma) + (1 / (2 * sigma ** 2)) * ((y - theta_1 - theta_2 * x) ** 2)
    
    return nll_i

# We'll now numerically approximate the gradient
def numeric_gradient(f, params, *args, h = 1.e-6):
    # initialize gradient as zeroes
    gradient = np.zeros(len(params))

    # We loop through each parameter
    for i in range(len(params)):
        param_change = params.copy()

        param_change[i] += h

        gradient[i] = (f(param_change, *args) - f(params, *args)) / h

    return gradient

# Now we have a function that, for any function f taking in arguments and parameters, calculates the gradient

# Now we use gradient descent to find the parameter which send the gradient (sufficiently close) to zero
def grad_descent(grad, f, init_param, *args, epsilon=1.e-6, eta=1.e-4, max_iter=1000):
    # Get our initial parameter guesses
    param_old = init_param.copy()

    # Start iteration count
    iteration = 0

    # Evaluate the gradient at this guess

    grad_old = grad(f, param_old, *args)

    # Now for as long as the gradient is sufficiently large, we loop through different parameter guesses

    while np.linalg.norm(grad_old) > epsilon and iteration < max_iter:
        param_guess = param_old - eta * grad_old

        grad_new = grad(f, param_guess, *args)

        param_old[:] = param_guess 
        grad_old[:] = grad_new

        iteration += 1

        print(f"Iteration{iteration}")

    return param_old


# Initial guesses for parameters
init_params = np.array([3.0, 3.0, 1.0])

results1b = grad_descent(numeric_gradient, normal_e_nll, init_params, Y1.flatten(), X1[:, 1])
print(f"The coefficient estimate for theta_1 is {results1b[0]}, for theta_2 is {results1b[1]}, and for sigma is {results1b[2]}")

# Now we need to calculate standard errors. This is given by the equation on the PSET

# We need to evaluate the gradient for each observation at the optimal parameters
opt_grad = np.zeros((len(df1), len(results1b)))

for i in range(len(results1b)):

    opt_params = results1b.copy()

    opt_params[i] += 1.e-6

    opt_grad[:, i] = (normal_e_nll_i(opt_params, Y1.flatten(), X1[:, 1]) - normal_e_nll_i(results1b, Y1.flatten(), X1[:, 1])) / 1.e-6

var_mle = np.linalg.inv(opt_grad.T @ opt_grad)

se_mle = np.sqrt(np.diag(var_mle))
print(f"The SE estimate for theta_1 is {se_mle[0]}, for theta_2 is {se_mle[1]}, and for sigma is {se_mle[2]}")
    
###################
#### 1.c ####
###################
    
# We follow Dan's problem setup. As Dan mentions, we can create an "optimal" weighting matrix. However, for practice we will first use the identity matrix.

# Given the model and our two moment conditions, we may write a function that calculates the sample moment counterparts

def stacked_moments(theta_1, theta_2, yvals, xvals):
        e_moment = yvals.mean() - theta_1 - theta_2 * xvals.mean()
        ex_moment = (xvals * yvals).mean() - xvals.mean() * theta_1 - (xvals ** 2).mean() * theta_2

        stacked_moments = np.array([[e_moment], [ex_moment]])

        return stacked_moments 

# We will use this function to calcualte the cov-var matrix 
def stacked_moments_i(theta_1, theta_2, yvals, xvals):
        e_moment = yvals - theta_1 - theta_2 * xvals
        ex_moment = (xvals * yvals) - xvals * theta_1 - (xvals ** 2) * theta_2

        stacked_moments = np.column_stack((e_moment, ex_moment))

        return stacked_moments 

# Next, we create the GMM objective function in quadratic form. Again, our first attempt will use the identity matrix

def gmm_obj(params, *args):
    theta_1 = params[0]
    theta_2 = params[1]

    yvals = args[0]
    xvals = args[1]
    A = args[2]

    G_N = stacked_moments(theta_1, theta_2, yvals, xvals)

    gmm_quad = G_N.T @ A @ G_N

    return gmm_quad

# Now we guess are initial parameters
init_params = np.array([0.5, 0.8])

# Our initial weighting matrix will be the identity
init_A = np.eye(2)

# We'll use the built-in minimize function
coefs_gmm_I = minimize(gmm_obj, init_params, args=(Y1, X1[:, [-1]], init_A))
print(f"The coefficient estimate for theta_1 is {coefs_gmm_I.x[0]} and for theta_2 is {coefs_gmm_I.x[1]}")

# We need to make sure our objects have the correct shapes
Y1 = Y1.reshape(-1,1)

g_i = stacked_moments_i(coefs_gmm_I.x[0], coefs_gmm_I.x[1], Y1, X1[:, [-1]])

g_i_diff = g_i - (1 / len(df1)) * np.sum(g_i, axis=0)
opt_A = np.linalg.inv((1 / len(df1)) * (1 / len(df1)) * (g_i_diff.T @ g_i_diff))

# We now can re-estimate the coefficients using the "optimal" matrix. Our estimates should not change
coefs_gmm_opt = minimize(gmm_obj, init_params, args=(Y1, X1[:, [-1]], opt_A))

print(f"The coefficient estimate for theta_1 is {coefs_gmm_opt.x[0]} and for theta_2 is {coefs_gmm_opt.x[1]}") # and they don't, as expected

# Now we want to compute the standard errors. We already have A and need to compute Gamma and V, using Dan's notation

# Gamma = np.array([[-1, -X1[:, -1].mean()], 
#                    [-X1[:, -1].mean(), -(X1[:, -1] ** 2).mean()]])

# Note the above is an analytic solution which we can derive given the functional form. Dan asks us to instead use numeric approximations for the Jacobian. This is very similar to our numeric gradient, only now we have a vector-valued function that is our stacked moment conditions. 

def numeric_jacobian(f, params, *args, h = 1.e-6):

    n = len(args)
    k = len(params)

    # create a blank Jacobian
    jacobian = np.zeros((n, k))

    for j in range(k): 
        param_change = params.copy()

        param_change[j] += h

        jacobian[:, [j]] = ((f(param_change[0], param_change[1], *args) - f(params[0], params[1], *args)) / h)

    return jacobian

Gamma = numeric_jacobian(stacked_moments, coefs_gmm_opt.x, Y1, X1[:, -1])

V = (1 / len(df1)) * (1 / len(df1)) * (g_i.T @ g_i)

# Thus we can simply follow Dan's provided formula 

np.linalg.inv(Gamma.T @ opt_A @ Gamma) @ (Gamma.T @ opt_A @ V @ opt_A @ Gamma) @ np.linalg.inv(Gamma.T @ opt_A @ Gamma)

# We can check if this is numerically equivalent to the second variance formula
cov_var = np.linalg.inv(Gamma) @ V @ np.linalg.inv(Gamma.T)

se_gmm_1 = np.sqrt(np.diagonal(cov_var))
print(f"The SE estimate for theta_1 is {se_gmm_1[0]} and for theta_2 is {se_gmm_1[1]}")
