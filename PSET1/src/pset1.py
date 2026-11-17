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

Gamma = numeric_jacobian(stacked_moments_b, best_guess_opt, Y2, X2)

V = (1 / len(df2)) * (1 / len(df2)) * (g_i.T @ g_i)

# We can check as in question 1 that the following two matrix operations produce the same output
np.linalg.inv(Gamma.T @ opt_A @ Gamma) @ (Gamma.T @ opt_A @ V @ opt_A @ Gamma) @ np.linalg.inv(Gamma.T @ opt_A @ Gamma)

cov_var = np.linalg.inv(Gamma) @ V @ np.linalg.inv(Gamma.T)

se_gmm_2 = np.sqrt(np.diagonal(cov_var))
print(f"The SE estimate for theta_1 is {se_gmm_2[0]} and for theta_2 is {se_gmm_2[1]}")

######################################
#### Question 3 ####
######################################

# let's load in the datasets

df3 = pd.read_csv(DATA / "data3.dat", sep = "\s\s", header = None)
draws = pd.read_csv(DATA / "drawsml.dat", sep = "\s\s", header = None) # rows correspond to markets and columns correspond to a draw for each

# set the column names based on Dan's ordering. each row corresponds to one of the 50 markets
df3.columns = ["p", "x1", "x2"]

p = df3["p"].to_numpy()
x3_1 = df3["x1"].to_numpy()
x3_2 = df3["x2"].to_numpy()

# 3.a has no coding work

###################
#### 3.b ####
###################

# In the solution write-up, I integrate out e_i^2. So, the draws Dan provided will be interpreted as e_i^2 draws for each market. 

def likelihood_i(params, draws, *args): 

    theta_1 = params[0]
    theta_2 = params[1]
    sigma = params[2]

    p = args[0]
    x1 = args[1]
    x2 = args[2]

    ll_vec = np.zeros(len(draws))

    # for each row i, corresponding to each market, we need the likelihood we observe the observed price given the model. 

    for i in range(len(draws)):

        mc_2 = np.exp(theta_1 + theta_2 * x2[i] + sigma * draws.iloc[i, :])
        
        ll_i_j = norm.pdf((1 / sigma) * (np.log(np.maximum(0.0001, 3 * p[i] - 100 - mc_2)) - theta_1 - theta_2 * x1[i])) * np.abs(3 / (sigma * np.maximum(0.0001, 3 * p[i] - 100 - mc_2)))

        # the probability is then just the average across the 20 simulations

        ll_vec[i] = (1 / len(draws.columns)) * np.sum(ll_i_j)

    return ll_vec

def log_ll(params, *args): 

    ll_i = likelihood_i(params, draws, *args)

    nlog_ll = -np.sum(np.log(ll_i))

    return nlog_ll

# now we solve for the parameters using gradient descent
init_params = np.array([0.9, 0.5, 0.5])

results3b = grad_descent(numeric_gradient, log_ll, init_params, p, x3_1, x3_2)
print(f"The coefficient estimate for theta_1 is {results3b[0]}, for theta_2 is {results3b[1]}, and for sigma is {results3b[2]}")

# Finally, we calculate the standard errors
opt_grad = np.zeros((len(df3), len(results3b)))

for i in range(len(results3b)):

    opt_params = results3b.copy()

    opt_params[i] += 1.e-6

    opt_grad[:, i] = (np.log(likelihood_i(opt_params, draws, p, x3_1, x3_2)) - np.log(likelihood_i(results3b, draws, p, x3_1, x3_2))) / 1.e-6

var_mle = np.linalg.inv(opt_grad.T @ opt_grad)

se_mle = np.sqrt(np.diag(var_mle))
print(f"The SE estimate for theta_1 is {se_mle[0]}, for theta_2 is {se_mle[1]}, and for sigma is {se_mle[2]}")

###################
#### 3.c ####
###################

# No coding work.

###################
#### 3.d ####
###################

draws_gmm = pd.read_csv(DATA / "drawsgmm.dat", sep = "\s\s", header = None)

# We'll follow the general GMM coding steps from questions 1 and 2, but now using simulated error terms from draws_gmm. We will however change the naming convention to order moments in the order Dan wrote them down. 

# Convert the relevant (simulated) errors into numpy arrays as with the observable columns. 
es_1 = draws_gmm.iloc[:, :20].to_numpy()
es_2 = draws_gmm.iloc[:, -20:].to_numpy()


def stacked_moments(theta_1, theta_2, sigma, pvals, x1vals, x2vals): 

    model_price = (1 / 3) * (100 + np.exp(theta_1 + theta_2 * x1vals[:, None] + sigma * es_1) + np.exp(theta_1 + theta_2 * x2vals[:, None] + sigma * es_2))

    m1 = pvals.mean() - model_price.mean()

    m2 = (pvals * x1vals).mean() - (x1vals[:, None] * model_price).mean()

    m3 = (pvals * x2vals).mean() - (x2vals[:, None] * model_price).mean()

    m4 = (pvals ** 2).mean() -  (model_price ** 2).mean()

    stacked_moments = np.array([[m1], [m2], [m3], [m4]])

    return stacked_moments

# This function will compute the moment condition for each market i, rather than the average condition across markets, which is done in the function above. We need this function to find the weighting matrix. 
def stacked_moments_i(theta_1, theta_2, sigma, pvals, x1vals, x2vals):

    model_price = (1 / 3) * (100 + np.exp(theta_1 + theta_2 * x1vals[:, None] + sigma * es_1) + np.exp(theta_1 + theta_2 * x2vals[:, None] + sigma * es_2))

    m1 = (pvals[:, None]  - model_price).mean(axis=1)[:, None]

    m2 = (pvals * x1vals)[:, None]  - (x1vals[:, None] * model_price).mean(axis=1)[:, None]

    m3 = (pvals * x2vals)[:, None] - (x2vals[:, None] * model_price).mean(axis=1)[:, None]

    m4 = (pvals ** 2)[:, None]  -  (model_price ** 2).mean(axis=1)[:, None]

    stacked_moments = np.column_stack((m1, m2, m3, m4))
    
    return stacked_moments 


# We need a new objective function to account for the three observables

def gmm_obj(params, *args): 
    theta_1 = params[0]
    theta_2 = params[1]
    sigma = params[2]

    pvals = args[0]
    x1vals = args[1]
    x2vals = args[2]
    A = args[3]

    G_N = stacked_moments(theta_1, theta_2, sigma, pvals, x1vals, x2vals)

    gmm_quad = G_N.T @ A @ G_N

    return gmm_quad

# Since there are more moment conditions than parameters, the model is over-identified. Following Dan's instruction, we need to choose a weight matrix A. Let's follow the naive guess that p = mc and regress to get the initial set of parameters. 

x = df3[['x1', 'x2']]
x = sm.add_constant(x)
y = np.log(df3[['p']])

naive_ols = sm.OLS(y, x).fit()
naive_guesses = np.array(naive_ols.params)

# We'll now input these initial parameter guesses into the moment conditions to get V

g_i_init = stacked_moments_i(naive_guesses[0], naive_guesses[1], naive_guesses[2], p, x3_1, x3_2)

N = len(df3)

# We compute V using Dan's de-meaned version in question 1
V = (1 / N) * (1 / N) * (g_i_init - g_i_init.mean(axis=0)).T @ (g_i_init - g_i_init.mean(axis=0))

A_init = np.linalg.inv(V)

# Now we can use minimize to solve. I place bounds on sigma because I found that without it the minimize function does not converge. I suspect this is because I am not correctly applying the naive OLS estimates correctly and thus my V matrix is wacky. 
coefs_gmm_init = minimize(gmm_obj, naive_guesses, args=(p, x3_1, x3_2, A_init), bounds = [(None, None), (None, None), (3.e-7, 10)])
print(f"The coefficient estimate for theta_1 is {coefs_gmm_init.x[0]}, for theta_2 is {coefs_gmm_init.x[1]}, and for sigma is {coefs_gmm_init.x[2]}")

# Next, we calculate the standard errors using the procedure in question 1.

def numeric_jacobian(f, params, *args, h = 1.e-6):

    n = 4 # four moment conditions
    k = len(params)

    # create a blank Jacobian
    jacobian = np.zeros((n, k))

    for j in range(k): 
        param_change = params.copy()

        param_change[j] += h

        jacobian[:, [j]] = ((f(param_change[0], param_change[1], param_change[2], *args) - f(params[0], params[1], params[2], *args)) / h)

    return jacobian

Gamma = numeric_jacobian(stacked_moments, coefs_gmm_init.x, p, x3_1, x3_2)

g_i_opt = stacked_moments_i(coefs_gmm_init.x[0], coefs_gmm_init.x[1], coefs_gmm_init.x[2], p, x3_1, x3_2)

# We compute V using Dan's de-meaned version in question 1
V = (1 / N) * (1 / N) * (g_i_opt).T @ (g_i_opt)

A_opt = np.linalg.inv(V)

# The asymptotic variance formula is then 
var_cov_matrix = np.linalg.inv(Gamma.T @ A_opt @ Gamma) @ (Gamma.T @ A_opt @ V @ A_opt @ Gamma) @ np.linalg.inv(Gamma.T @ A_opt @ Gamma)

# Standard errors are then

se_gmm_3 = np.sqrt(np.diag(var_cov_matrix))

print(f"The SE estimate for theta_1 is {se_gmm_3[0]}, for theta_2 is {se_gmm_3[1]}, and for sigma is {se_gmm_3[2]}")
