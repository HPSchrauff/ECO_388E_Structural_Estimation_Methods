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
from statsmodels.discrete.discrete_model import Probit
from statsmodels.discrete.discrete_model import Logit
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import root

MAIN = Path(__file__).resolve().parents[1]

DATA = MAIN / "data"
CODE = MAIN / "src"
OUTPUT = MAIN / "output"

for directory in (DATA, CODE, OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(20)

######################################
#### Question 1 ####
######################################

###################
#### 1.a ####
###################

ps2_dat = pd.read_csv(
    DATA / "ps2.dat", 
    sep = "\s\s", 
    header = 0, 
    names = ['labor_force_dummy', 'age', 'educ', 'mean_parental_educ'],
    engine="python"
)

Y = ps2_dat['labor_force_dummy']
X = ps2_dat[['age', 'educ']]
X = sm.add_constant(X).astype(float)

# Let's use the built-in probit command for reference
probit = Probit(Y, X)
model_probit = probit.fit()
print(model_probit.summary())

def probit_likelihood(params, *args):
    theta_0, theta_1, theta_2 = params
    y, x1, x2 = args

    z = theta_0 + theta_1 * x1 + theta_2 * x2

    ll = -np.sum(
        y * norm.logcdf(z)
        + (1 - y) * norm.logcdf(-z)
    )

    return ll

def probit_likelihood_i(params, *args):
    theta_0, theta_1, theta_2 = params
    y, x1, x2 = args

    z = theta_0 + theta_1 * x1 + theta_2 * x2

    ll_i = (
        y * norm.logcdf(z)
        + (1 - y) * norm.logcdf(-z)
    )

    return ll_i

init_params = np.array([0.0, 0.0, 0.0])

probit_ml = minimize(
    probit_likelihood, 
    init_params, 
    args=(Y, X['age'], X['educ']),
    tol=1e-12)

i_ll = probit_likelihood_i(probit_ml.x, Y, X['age'], X['educ'])

# Now we want to calculate the standard errors, which in turn requires us to calculate the variance-covariance matrix.
def gradient(f, params, *args, h=1.e-6):

    gradient = np.zeros((len(ps2_dat), len(params)))

    for i in range(len(params)): 
        param_plus = params.copy()
        params_minus = params.copy()

        param_plus[i] += h 
        params_minus[i] -= h

        gradient[:, i] = (f(param_plus, *args) - f(params_minus, *args)) / (2 * h)

    return gradient

var_mle = np.linalg.inv(
    gradient(probit_likelihood_i, probit_ml.x, Y, X['age'], X['educ']).T
    @ 
    gradient(probit_likelihood_i, probit_ml.x, Y, X['age'], X['educ'])
)

se_mle = np.sqrt(np.diag(var_mle))

rows = []
for est, se in zip(probit_ml.x, se_mle):
    rows += [f"{est:.3f}", f"({se:.3f})"]

table_1a = pd.DataFrame(
    {"Probit": rows},
    index=["Constant", "", "Age", "", "Education", ""],
)

table_1a.to_latex(OUTPUT / "table_1a_probit.tex", column_format="lc")

###################
#### 1.b ####
###################

# We want to know the effect of an additional year of education on the probability of an average woman working. So, we need to get the average age and education
avg_age = ps2_dat['age'].mean()
avg_educ = ps2_dat['educ'].mean()

increase_prob = norm.cdf(probit_ml.x[0] + avg_age * probit_ml.x[1] + (1 + avg_educ) * probit_ml.x[2]) - norm.cdf(probit_ml.x[0] + avg_age * probit_ml.x[1] + (avg_educ) * probit_ml.x[2])

print(increase_prob)

###################
#### 1.c ####
###################

# We'll first use the built-in

logit = Logit(Y, X)
model_logit = logit.fit()
print(model_logit.summary())

def logit_likelihood(params, *args): 
    theta_0, theta_1, theta_2 = params
    y, x1, x2 = args 

    z = theta_0 + theta_1 * x1 + theta_2 * x2

    neg_ll = - np.sum(
        y * -np.logaddexp(0, -z) + 
        (1 - y) * -np.logaddexp(0, z)
    )

    return neg_ll

init_params = np.array([0.1, 0.1, 0.1])

#logit_ml = minimize(
#    logit_likelihood,
#    init_params,
#    args=(Y, X['age'], X['educ']),
#    tol=1e-12
#) # this method isn't converging. So, let's try gradient descent with a numeric gradient

def gradient_ll(f, params, *args, h=1.e-6):

    gradient = np.zeros(len(params))

    for i in range(len(params)): 
        param_plus = params.copy()
        params_minus = params.copy()

        param_plus[i] += h 
        params_minus[i] -= h

        gradient[i] = (f(param_plus, *args) - f(params_minus, *args)) / (2 * h)

    return gradient

def grad_descent(grad, f, init_params, *args, epsilon=1.e-6, eta=1.e-6, max_iter=10000):

    param_old = init_params.copy()
    iteration = 0
    grad_old = grad(f, param_old, *args)

    while np.linalg.norm(grad_old) > epsilon and iteration < max_iter: 
        param_guess = param_old - eta * grad_old 

        grad_new = grad(f, param_guess, *args)

        param_old[:] = param_guess
        grad_old[:] = grad_new 

        iteration += 1

        print(f"Iteration{iteration}")

    return param_old

#logit_grad_ml = grad_descent(
#    gradient_ll, 
#    logit_likelihood, 
#    init_params, 
#    Y, X['age'], X['educ']
#) # this also doesn't work. analytical gradient it is. 

def logit_score(params, y, x1, x2):
    z = params[0] + params[1] * x1 + params[2] * x2
    foc = y - (np.exp(z) / (1 + np.exp(z)))
    return np.array([foc.sum(), (foc * x1).sum(), (foc * x2).sum()])

logit_root = root(logit_score, init_params, args=(Y, X['age'], X['educ']))

# Next, we need standard errors. 

logit_theta_hats = logit_root.x

z_hat = X @ logit_theta_hats

W = pd.DataFrame(
    np.diag((1 / (1 + np.exp(z_hat))) * (1 - (1 / (1 + (np.exp(z_hat))))))
)

cov_logit = np.linalg.inv(X.T @ W @ X)

se_mle_logit = np.sqrt(np.diag(cov_logit))

rows = []
for est, se in zip(logit_theta_hats, se_mle_logit):
    rows += [f"{est:.3f}", f"({se:.3f})"]

table_1c = pd.DataFrame(
    {"Logit": rows},
    index=["Constant", "", "Age", "", "Education", ""],
)

table_1c.to_latex(OUTPUT / "table_1c_logit.tex", column_format="lc")

logit_ratio = logit_theta_hats[1] / logit_theta_hats[2]
probit_ratio = probit_ml.x[1] / probit_ml.x[2]