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
#### Question 2 ####
######################################

###################
#### 2.a ####
###################

df2 = pd.read_csv(DATA / "data2.dat", sep = "\s\s", header = None)

df2["cons"] = 1
df2.columns = ["y", "x", "cons"]

# We don't estimate OLS, so we don't need to include the constant column in X2
Y2 = df2["y"].to_numpy()
X2 = df2["x"].to_numpy()

# We'll follow the general outline we used in question 1
def exp_nll(params, *args): 
    n = len(df2)

    theta_1 = params[0]
    theta_2 = params[1]
    sigma = params[2]
    
    y = args[0]
    x = args[1]

    nll = n * np.log(sigma) + 0.5 * n * np.log(2 * pi) + np.sum(np.log(y)  + (1 / (2 * sigma ** 2)) * (np.log(y) - theta_1 - x ** theta_2) ** 2)

    return nll

def exp_nll_i(params, *args): 

    theta_1 = params[0]
    theta_2 = params[1]
    sigma = params[2]
    
    y = args[0]
    x = args[1]

    nll = np.log(sigma) + 0.5 * np.log(2 * pi) + np.log(y)  + (1 / (2 * sigma ** 2)) * (np.log(y) - theta_1 - x ** theta_2) ** 2

    return nll

# we already have the numeric_gradient function to numerically approximate the gradient if we didn't have an analytic solution. What changes here is we need to try several different starting values and pick the parameters which achieve the maximum across our different starting guesses. 

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

def numeric_gradient(f, params, *args, h = 1.e-6):
    # initialize gradient as zeroes
    gradient = np.zeros(len(params))

    # We loop through each parameter
    for i in range(len(params)):
        param_change = params.copy()

        param_change[i] += h

        gradient[i] = (f(param_change, *args) - f(params, *args)) / h

    return gradient

n_guesses = 50

init_guesses = np.random.uniform(low=1.e-6, high=10, size=(n_guesses, 3))
ll_results = np.zeros((n_guesses, 1))
ll_argmin = np.zeros((n_guesses, 3))

for i in range(len(init_guesses)): 
    init_params = init_guesses[i]

    results = grad_descent(numeric_gradient, exp_nll, init_params, Y2, X2)

    ll_argmin[i] = results

    ll_results[i] = exp_nll(results, Y2, X2)

    print(f"Iteration {i + 1} complete")

max_ll = np.argmin(ll_results)
best_guess = np.round(ll_argmin[max_ll], decimals = 4)
print(f"The coefficient estimates for theta_1 is {best_guess[0]}, for theta_2 is {best_guess[1]}, and for sigma is {best_guess[2]}")

opt_grad = np.zeros((len(df2), len(best_guess)))

for i in range(len(best_guess)):

    opt_params = best_guess.copy()

    opt_params[i] += 1.e-6

    opt_grad[:, i] = (exp_nll_i(opt_params, Y2, X2) - exp_nll_i(best_guess, Y2, X2)) / 1.e-6

var_mle = np.linalg.inv(opt_grad.T @ opt_grad)

se_mle = np.round(np.sqrt(np.diag(var_mle)), decimals = 4)
print(f"The SE for theta_1 is {se_mle[0]}, for theta_2 is {se_mle[1]}, and for sigma is {se_mle[2]}")

###################
#### 2.b ####
###################

# Given the model, we need to define different stacked moments

def stacked_moments_b(theta_1, theta_2, yvals, xvals): 
    e_moment = np.log(yvals).mean() - theta_1 - (xvals ** theta_2).mean()
    ex_moment = (xvals * np.log(yvals)).mean() - xvals.mean() * theta_1 - (xvals ** (theta_2 + 1)).mean()

    stacked_moments = np.array([[e_moment], [ex_moment]])

    return stacked_moments

def stacked_moments_i_b(theta_1, theta_2, yvals, xvals): 
    e_moment = np.log(yvals) - theta_1 - xvals ** theta_2 
    ex_moment = xvals * np.log(yvals) - xvals * theta_1 - xvals ** (theta_2 + 1)

    stacked_moments = np.column_stack((e_moment, ex_moment))
    
    return stacked_moments 

# We re-write the GMM objective function below for practice

def gmm_obj(params, *args):
    theta_1 = params[0]
    theta_2 = params[1]

    yvals = args[0]
    xvals = args[1]
    A = args[2]

    G_N = stacked_moments_b(theta_1, theta_2, yvals, xvals)

    gmm_quad = G_N.T @ A @ G_N

    return gmm_quad

# Let's try with one guess
init_params = np.array([1.0, 0.8])
init_A = np.eye(2)

coefs_gmm_I = minimize(gmm_obj, init_params, args=(Y2, X2, init_A))
print(f"The coefficient estimate for theta_1 is {coefs_gmm_I.x[0]} and for theta_2 is {coefs_gmm_I.x[1]}")


# Following Dan's suggestion, we'll look for a few different starting values. In this case we'll select the estimates that produce the small gmm_quad objective function

n_guesses = 50
init_guesses = np.random.uniform(low=1.e-6, high=10, size=(n_guesses, 2))
gmm_results = np.zeros((n_guesses, 1))
gmm_argmin = np.zeros((n_guesses, 2))

# We'll start with the identity matrix before finding the optimal weighting matrix
init_A = np.eye(2)

for i in range(len(init_guesses)): 
    init_params = init_guesses[i]

    results = minimize(gmm_obj, init_params, args = (Y2, X2, init_A))

    gmm_argmin[i] = results.x
    
    gmm_results[i] = results.fun

    print(f"Iteration {i + 1} complete")

min_gmm_obj = np.argmin(gmm_results)
best_guess = np.round(gmm_argmin[min_gmm_obj], decimals=4)
print(f"The coefficient estimate for theta_1 is {best_guess[0]} and for theta_2 is {best_guess[1]}")
# The point estimate matches for theta_1 but differs for theta_2

# Let's get the optimal weighting matrix
g_i = stacked_moments_i_b(best_guess[0], best_guess[1], Y2, X2)
g_i_diff = g_i - (1 / len(df2)) * np.sum(g_i, axis=0)
opt_A = np.linalg.inv((1 / len(df2)) * (1 / len(df2)) * (g_i_diff.T @ g_i_diff))

gmm_results_opt = np.zeros((n_guesses, 1))
gmm_argmin_opt = np.zeros((n_guesses, 2))

for i in range(len(init_guesses)): 
    init_params = init_guesses[i]

    results_opt = minimize(gmm_obj, init_params, args = (Y2, X2, opt_A))

    gmm_argmin_opt[i] = results_opt.x
    
    gmm_results_opt[i] = results_opt.fun

    print(f"Iteration {i + 1} complete")

min_gmm_obj_opt = np.argmin(gmm_results_opt)
best_guess_opt = np.round(gmm_argmin_opt[min_gmm_obj_opt], decimals=4)
print(f"The coefficient estimate for theta_1 is {best_guess_opt[0]} and for theta_2 is {best_guess_opt[1]}")

# As expected, since the model is exactly identified the coefficient estimates do not change. 
# We'll next estimate the standard errors. 

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

Gamma = numeric_jacobian(stacked_moments_b, best_guess_opt, Y2, X2)

V = (1 / len(df2)) * (1 / len(df2)) * (g_i.T @ g_i)

# We can check as in question 1 that the following two matrix operations produce the same output
np.linalg.inv(Gamma.T @ opt_A @ Gamma) @ (Gamma.T @ opt_A @ V @ opt_A @ Gamma) @ np.linalg.inv(Gamma.T @ opt_A @ Gamma)

cov_var = np.linalg.inv(Gamma) @ V @ np.linalg.inv(Gamma.T)

se_gmm_2 = np.sqrt(np.diagonal(cov_var))
print(f"The SE estimate for theta_1 is {se_gmm_2[0]} and for theta_2 is {se_gmm_2[1]}")