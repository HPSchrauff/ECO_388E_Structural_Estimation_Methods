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

def numeric_gradient(f, params, *args, h = 1.e-6):
    # initialize gradient as zeroes
    gradient = np.zeros(len(params))

    # We loop through each parameter
    for i in range(len(params)):
        param_change = params.copy()

        param_change[i] += h

        gradient[i] = (f(param_change, *args) - f(params, *args)) / h

    return gradient

def grad_descent(grad, f, init_param, *args, epsilon=1.e-6, eta=1.e-4, max_iter=10):
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

###################
#### 3.e ####
###################
# Now we will not use the last moment condition. 

def stacked_moments(theta_1, theta_2, sigma, pvals, x1vals, x2vals): 

    model_price = (1 / 3) * (100 + np.exp(theta_1 + theta_2 * x1vals[:, None] + sigma * es_1) + np.exp(theta_1 + theta_2 * x2vals[:, None] + sigma * es_2))

    m1 = pvals.mean() - model_price.mean()

    m2 = (pvals * x1vals).mean() - (x1vals[:, None] * model_price).mean()

    m3 = (pvals * x2vals).mean() - (x2vals[:, None] * model_price).mean()

    stacked_moments = np.array([[m1], [m2], [m3]])

    return stacked_moments

def stacked_moments_i(theta_1, theta_2, sigma, pvals, x1vals, x2vals):

    model_price = (1 / 3) * (100 + np.exp(theta_1 + theta_2 * x1vals[:, None] + sigma * es_1) + np.exp(theta_1 + theta_2 * x2vals[:, None] + sigma * es_2))

    m1 = (pvals[:, None]  - model_price).mean(axis=1)[:, None]

    m2 = (pvals * x1vals)[:, None]  - (x1vals[:, None] * model_price).mean(axis=1)[:, None]

    m3 = (pvals * x2vals)[:, None] - (x2vals[:, None] * model_price).mean(axis=1)[:, None]

    stacked_moments = np.column_stack((m1, m2, m3))
    
    return stacked_moments 


g_i_init = stacked_moments_i(naive_guesses[0], naive_guesses[1], naive_guesses[2], p, x3_1, x3_2)

V = (1 / N) * (1 / N) * (g_i_init - g_i_init.mean(axis=0)).T @ (g_i_init - g_i_init.mean(axis=0))

A_init = np.linalg.inv(V)

coefs_gmm_init = minimize(gmm_obj, naive_guesses, args=(p, x3_1, x3_2, A_init), bounds = [(None, None), (None, None), (3.e-7, 10)])
print(f"The coefficient estimate for theta_1 is {coefs_gmm_init.x[0]}, for theta_2 is {coefs_gmm_init.x[1]}, and for sigma is {coefs_gmm_init.x[2]}")

def numeric_jacobian(f, params, *args, h = 1.e-6):

    n = 3 # three moment conditions
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

V = (1 / N) * (1 / N) * (g_i_opt).T @ (g_i_opt)

A_opt = np.linalg.inv(V)

var_cov_matrix = np.linalg.inv(Gamma.T @ A_opt @ Gamma) @ (Gamma.T @ A_opt @ V @ A_opt @ Gamma) @ np.linalg.inv(Gamma.T @ A_opt @ Gamma)

se_gmm_3 = np.sqrt(np.diag(var_cov_matrix))

print(f"The SE estimate for theta_1 is {se_gmm_3[0]}, for theta_2 is {se_gmm_3[1]}, and for sigma is {se_gmm_3[2]}")