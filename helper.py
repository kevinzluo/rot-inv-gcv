import numpy as np
from scipy.stats import ortho_group
from tqdm import tqdm
import pandas as pd

def ridge_solution(features, response, reg = 0):
    return np.linalg.pinv(features.T @ features + reg * np.eye(features.shape[1])) @ features.T @ response

def construct_diagonal(d, n, p):
    D = np.zeros((n, p))

    assert len(d) == n or len(d) == p

    l = np.min([n, p])
    for i in range(l):
        D[i, i] = d[i]
    return D

def riri_sample(d, n, p, Q = None, return_svd = False):
    if Q is None:
        Q = ortho_group.rvs(dim = n)

    assert Q.shape[0] == Q.shape[1] == n
    D = construct_diagonal(d, n, p)
    
    O = ortho_group.rvs(dim=p)
    if not return_svd:
        return Q.T @ D @ O, np.array(d, dtype = float)
    # return full
    return Q.T @ D @ O, Q.T, np.array(d, dtype = float), O

def gaussian_sample(n, p, return_svd = False):
    X = np.random.normal(size = (n, p)) / np.sqrt(n)

    if not return_svd:
        d = np.linalg.svd(X, compute_uv=False)

        return X, d
    
    Qt, d, O = np.linalg.svd(X)
    return X, Qt, d, O

def gaussian_dupe_sample(n, p, frac_to_double = 0.3, return_svd = False):
    base_samples = int(n / (1 + frac_to_double))
    X = np.random.normal(size = (base_samples, p)) / np.sqrt(n)
    duplicate = np.random.choice(base_samples, size = n - base_samples, replace = False)
    X_full = np.concatenate([X, X[duplicate]])

    if not return_svd:
        return X_full
    else:
        Qt, d, O = np.linalg.svd(X_full)
        # print(np.sum(d ** 2))

        return X_full, Qt, d, O

def gaussian_mixture_sample(n, p, mu = 1, return_svd = False):
    X = np.random.normal(size = (n, p))
    means = np.random.choice(2, size = (n))
    means = (means * 2 - 1) * mu
    X += means[:, None]

    X /= np.sqrt(1 + mu ** 2) * np.sqrt(n) # get back down to [0, 1] entries

    if not return_svd:
        d = np.linalg.svd(X, compute_uv=False)

        return X, d
    
    Qt, d, O = np.linalg.svd(X)
    return X, Qt, d, O

def gaussian_row_corr_sample(n, p, rho = 0.6, return_svd = False):
    # X = np.random.normal(size = (n, p))
    Sigma = np.ones((p, p))
    rho = 0.75
    for i in range(p):
        for j in range(p):
            if i != j:
                Sigma[i, j] = rho # 0.7 ** np.abs(i - j)
    Z = np.random.normal(size = (n, p)) 
    sig_evals, sig_evecs = np.linalg.eigh(Sigma)
    sig_evals = np.real(sig_evals)
    sig_evecs = np.real(sig_evecs)
    Sigma_sqrt = sig_evecs @ np.diag(np.sqrt(sig_evals)) @ sig_evecs.T

    # np.linalg.sqrtm()
    X = Z @ Sigma_sqrt / np.sqrt(n)

    if not return_svd:
        d = np.linalg.svd(X, compute_uv=False)

        return X, d
    
    Qt, d, O = np.linalg.svd(X)
    return X, Qt, d, O



def gaussian_noise(n, sigma):
    return np.random.normal(scale = sigma, size = (n,))

def lnn_sample(n, p, k = 4, return_svd = False):
    # sample intermediate dimensions
    to_sample = k - 1


    # diff = abs(p - n)
    # lesser = min(n, p)
    # intermediate_dims = np.random.choice(diff + 1, size = to_sample, replace = True) + lesser
    intermediate_dims = np.linspace(n, p, k, dtype = int)

    dims = [n] + list(intermediate_dims) + [p]
    prod = np.eye(n)
    for i in range(k):
        X_new = np.random.normal(size = (dims[i], dims[i + 1]))
        prod = prod @ X_new
    
    # normalize product
    tr = np.trace(prod.T @ prod)
    factor = tr / p

    out_matrix = prod / np.sqrt(factor)
    # print(out_matrix.shape)

    if not return_svd:
        return out_matrix
    else:
        Qt, d, O = np.linalg.svd(out_matrix)
        # print(np.sum(d ** 2))

        return out_matrix, Qt, d, O

def rho_auto_sample(n, p, rho = 0.3, return_svd = False):
    X = np.zeros((n, p))
    X[0] = np.random.normal(size = p)
    for i in range(n - 1):
        X[i + 1] = X[i] * rho + np.random.normal(size = p) * np.sqrt(1 - rho ** 2)
    # tr = np.trace(X.T @ X)
    # factor = tr / p
    factor = 1

    X = X / np.sqrt(n)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    



from scipy.stats import matrix_normal, invwishart

def fat_rho_auto_sample(n, p, rho = 0.3, return_svd = False):
    X = np.zeros((n, p))
    X[0] = np.random.normal(size = p)

    Sigma = invwishart(int(1.1 * p), np.eye(p) * int(p * 1.1))

    for i in range(n - 1):
        X[i + 1] = X[i] * rho + np.random.normal(size = p) * np.sqrt(1 - rho ** 2)
    
    X = X / np.sqrt(n)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    


def matrix_normal_sample(n, p, rho = 0.5, return_svd = False):
    colcov = invwishart(int(1.1 * p), np.eye(p) * int(p * 1.1) ).rvs()
    rowcov = rho ** np.abs(np.arange(n) - np.arange(n)[:, None])
    # rowcov = invwishart(int(1.002 * n), np.eye(n) * int(n * 1.002) ).rvs()
    # rowcov = invwishart(int(1.1 * n), np.eye(n)).rvs()

    X = matrix_normal(np.zeros((n, p)), rowcov = rowcov, colcov = colcov).rvs()

    print(X.shape)

    print(np.trace(X.T @ X))

    X = X / np.sqrt(n)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    




from scipy.stats import multivariate_t, pareto
def t_sample(n, p, scale = 1/3, dof = 3, return_svd = False, standardize = False):
    t = multivariate_t(loc = None, shape = np.eye(p) * scale, df = dof,)
    X = t.rvs(size = n)

    if standardize:
        X /= X.std(0)

    X = X / np.sqrt(n)


    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    

def pareto_sample(n, p, alpha = 2.1, return_svd = False,):
    rescale = np.sqrt(alpha / (alpha - 1) ** 2 / (alpha - 2))
    pareto_sampler = pareto(alpha, loc = -1 * alpha / (alpha - 1) / rescale, scale = 1 / rescale)
    X = pareto_sampler.rvs(size = (n, p))

    X = X / np.sqrt(n)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    

def latent_base_sample(n, p, rho = 0.1, return_svd = False):
    X = np.random.normal(size = (n, p))
    latent = np.random.normal(size = p)
    X = X * np.sqrt(1 - rho ** 2) + latent * rho

    X = X / np.sqrt(n)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    

def rademacher_sample(n, p, return_svd = False):
    X = np.random.choice([-1, 1], size = (n, p))

    X = X / np.sqrt(n)
    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    

def uniform_sample(n, p, return_svd = False):
    X = np.random.uniform(low = -np.sqrt(12) / 2, high = np.sqrt(12)/2, size = (n, p))

    X = X / np.sqrt(n)
    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    

def var_sample(n, p, alphas = None, return_svd = False):
    if alphas is None:
        alphas = [0.4, 0.08, 0.04]


    X = np.zeros((n, p))
    for i in range(n - 1):
        for k in range(min(len(alphas), i + 1)):
            X[i + 1] = X[i + 1 - k] * alphas[k]
        X[i + 1] += np.random.normal(size = p)
    
    tr = np.trace(X.T @ X)
    factor = tr / p

    X = X / np.sqrt(factor)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    



def spiked_sample(n, p, k = 50, alpha = 10, return_svd = False):
    noise = np.random.normal(size = (n, p)) / np.sqrt(n)
    V = ortho_group.rvs(dim = n)[:k] # k x n
    W = ortho_group.rvs(dim = p)[:k] # k x p

    X = alpha * V.T @ W + noise

    tr = np.trace(X.T @ X)
    factor = tr / p

    X = X / np.sqrt(factor)

    if not return_svd:
        return X
    else:
        Qt, d, O = np.linalg.svd(X)
        return X, Qt, d, O    

    

def beta_sampler(p, r = None):
    if r is None:
        normalize = False
    else:
        normalize = True

    b = np.random.normal(size = p)
    if normalize:
        b /= (b ** 2).sum() ** 0.5
        b *= r
    return b

def signed_beta_sampler(p, r = None):
    if r is None:
        normalize = False
    else:
        normalize = True

    b = np.random.choice(2, size=p) * 2 - 1
    b = b.astype(float)
    if normalize:
        b /= (b ** 2).sum() ** 0.5
        b *= r
    return b

def dumb_variance(X, beta, noise_sampler, samples = 1000):
    beta_hats = np.zeros((samples, X.shape[1]))

    for i in range(samples):
        y = X @ beta + noise_sampler()
        beta_hats[i] = ridge_solution(X, y,)
    
    print(beta_hats.shape)
    print(np.cov(beta_hats).shape)
    print(np.cov(beta_hats.T))
    print(np.linalg.pinv(X.T @ X))
    
    n = X.shape[0]
    return 1/n * np.trace(np.cov(beta_hats, rowvar=False))


def linear_spectrum(d):
    return np.arange(1, d + 1) / d

# a theoretical quantity is passed the following data within the inner loop:
# X, d, beta
# r and sigma should be presets external to the risk estimation

## ridgeless stuff
def ridgeless_exp_bias(X, r, d, dims = None):
    if dims is None:
        n, p = X.shape
    else:
        n,p = dims
    # print(n,p)

    # tr_sq = np.sum(d ** 2)
    
    return r ** 2 * (1 - min(n, p) / p) # * (tr_sq / p)

def ridgeless_variance(X, sigma, d, dims = None):
    if dims is None:
        n, p = X.shape
    else:
        n,p = dims

    # so we get destroyed by numerical instability here
    # pinv = np.linalg.pinv(X.T @ X)
    # print("eigenvalues of inv", np.linalg.eigvals(pinv))
    # tr_inv = np.trace(np.linalg.pinv(X.T @ X))
    # print("trace inv squared", tr_inv)
    # tr_sq = np.trace(X.T @ X)
    # print("trace squared", tr_sq)

    # print(d)

    tr_sq = np.sum(d ** 2)
    # print(tr_sq / p)

    # maybe need some thresholding here
    tr_inv = np.sum(d[d != 0] ** (-2))
    # print(tr_inv / n)

    # print("unscaled variance", sigma ** 2 / (n) * tr_inv )
    # print("additional scale", tr_sq / p)

    return sigma ** 2 / (n) * tr_inv # * tr_sq / p

def ridgeless_risk(X, sigma, r, d, dims = None):
    # print(ridgeless_bias(X, r, d))
    # print(ridgeless_variance(X, sigma, d))
    return ridgeless_exp_bias(X, r, d, dims = dims) + ridgeless_variance(X, sigma, d, dims = dims)

def fast_ridgeless_exact_bias(DtD_inv, D, O, beta,):
    # n, p = X.shape
    # return 1/n * beta.T @ (np.eye(p) - np.linalg.pinv(X.T @ X) @ X.T @ X) @ beta # * np.sum(d ** 2) / p

    n,p = D.shape
    return 1/n * ( (beta**2).sum() - beta.T @ O.T @ DtD_inv @ D.T @ D @ O @ beta )

def ridgeless_exact_bias(X, d, beta,):
    n, p = X.shape
    return 1/n * beta.T @ (np.eye(p) - np.linalg.pinv(X.T @ X) @ X.T @ X) @ beta # * np.sum(d ** 2) / p
    
    # diagonal = [1] * min(n, p)
    # if n < p:
    #     diagonal += [0] * (p - n)
    # diagonal = np.array(diagonal)

    # _, _, O = np.linalg.svd(X)

    # return 1/n * beta.T @ O.T @ (np.eye(p) - np.diag(diagonal)) @ O @ beta * np.sum(d ** 2) / p

def ridgeless_exact_variance(X, d, sigma):
    n, p = X.shape
    return sigma ** 2 / n * np.sum( d ** -2 ) # * np.sum(d ** 2) / p

## ridge stuff
def ridge_exp_bias(X, d, beta, reg, dims = None):
    # print(reg)
    if dims is None:
        n, p = X.shape
    else:
        n, p = dims

    r2 = (beta ** 2).sum() / n

    adjustment = p - n if p > n else 0

    return r2 * (np.sum( (1 - (d ** 2 + reg) ** -1 * d ** 2) ** 2 ) + adjustment) / p # * np.sum(d ** 2) / p

def alt_ridge_exp_bias(DtD_inv, D, O, beta):
    n, p = D.shape

    middle = np.eye(p) - DtD_inv @ D.T @ D
    r2 = (beta ** 2).sum() / n

    return np.trace(middle @ middle) * r2 / p


def ridge_variance(X, sigma, d, reg):
    n, p = X.shape

    # print(sigma, d, reg)

    return sigma ** 2 / n * np.sum( (d ** 2 + reg) ** -2 * d ** 2 )#  * np.sum(d ** 2) / p

def fast_ridge_exact_bias(DtD_inv, D, O, beta):
    n, p = D.shape

    middle = np.eye(p) - DtD_inv @ D.T @ D

    return 1/n * beta.T @ O.T @ middle @ middle @ O @ beta

def ridge_exact_bias(X, d, beta, reg,):
    n, p = X.shape

    M = np.eye(p) - np.linalg.pinv(X.T @ X + reg * np.eye(p)) @ X.T @ X
    return beta.T @ M @ M @ beta / (n)#  * np.sum(d ** 2) / p

## gcv stuff
def gcv(regs, x_sampler, beta_sampler, noise_sampler, n_iters, x_iters = 1, seed = None):
    if seed is not None:
        np.random.seed(seed)
    tracked_estimators = {}
    for ind in range(x_iters):
        tracked_estimators[ind] = {}

        for k, e in enumerate(regs):
            tracked_estimators[ind][k] = []

        for _ in tqdm(range(n_iters)):
            X, Qt, d, O = x_sampler(return_svd = True)
            n, p = X.shape
            d_full = list(d) + [0] * ( max(n, p) - n )
            d_full = np.array(d_full)
            
            beta = beta_sampler()
            y = X @ beta + noise_sampler()

            for k, r in enumerate(regs):
                # fitted = ridge_solution(X, y, reg = r)
                # yhat = X @ fitted

                inner_matrix_diag = d ** 2 / (d ** 2 + r)

                inner_matrix = np.zeros((n,n))
                np.fill_diagonal(inner_matrix, inner_matrix_diag)

                # S_lam = X @ np.linalg.pinv(X.T @ X + r * np.eye(p)) @ X.T
                # print(np.trace(S_lam) / n)
                # print(S_lam[0])
                S_lam = Qt @ inner_matrix @ Qt.T
                # print(S_lam[0])
                tr = np.trace(S_lam)
                # print(tr / n)

                # denom_correction = (np.sum( (d_full ** 2 + r) ** -1 ) / p) ** -1 - r
                # print("denom_correction", denom_correction)

                # old_correction = (1 - tr / n)
                # print("old correction", old_correction)

                # tracked_estimators[ind][k].append(
                #     y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y / n / (denom_correction ** 2) # (1 - tr / n) ** 2
                # )
                tracked_estimators[ind][k].append(
                    y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y / n / (1 - tr / n) ** 2
                )
                # tracked_estimators[ind][k].append(
                #     np.mean(
                #         (y - yhat) ** 2
                #     ) / (1 - tr / n) ** 2
                # )

    return tracked_estimators

def l2_loss(y1, y2):
    return ((y2 - y1) ** 2).mean()

def track_risk_ensemble(estimators, theoreticals, n_iters, x_sampler, beta_sampler, noise_sampler, x_iters = 1, loss_func = None, seed = None):
    if loss_func is None:
        loss_func = l2_loss

    if seed is not None:
        np.random.seed(seed)

    trained_estimators = {}
    tracked_theoreticals = {}
    tracked_asymptotics = {}
    estimator_losses = {}

    for x_iter in range(x_iters):
        estimator_losses[x_iter] = {}
        tracked_theoreticals[x_iter] = []
        # print("generating data")
        X, d = x_sampler()
        # print('d')
        # print(d)
        beta = beta_sampler()
        # print("beta")
        # print(beta)

        # print(np.sum( d ** -2 ))
        # print(np.sum(d ** 2))
        # print(beta)

        # print("iterating over n_iters")
        # so it turns out this has serious problems with high dimensions
        # who would've thought 🙄
        # so actually we should be using the analytical expressions for the bias and variance you dumbass

        for _ in range(n_iters):
            
            y = X @ beta + noise_sampler()
            X_new, _ = x_sampler()

            for k, e in enumerate(estimators):
                if k not in estimator_losses[x_iter].keys():
                    estimator_losses[x_iter][k] = []

                beta_hat = e(X, y)
                estimator_losses[x_iter][k].append(
                    loss_func(X_new @ beta_hat, X_new @ beta)
                )

        for t in theoreticals:
            tracked_theoreticals[x_iter].append(
                t(X, d, beta)
            )
        
    return estimator_losses, tracked_theoreticals

def fast_track_risk_ensemble(regs, n_iters, x_sampler, beta_sampler, noise_sampler, r = 1, sigma = 1, x_iters = 1, loss_func = None, seed = None):
    if loss_func is None:
        loss_func = l2_loss

    if seed is not None:
        np.random.seed(seed)

    tracked_theoreticals = {}
    estimator_losses = {}

    for x_iter in range(x_iters):
        estimator_losses[x_iter] = {}
        tracked_theoreticals[x_iter] = {}
        # print("generating data")
        X, Qt, d, O = x_sampler(return_svd = True)
        n, p = X.shape

        d_aug = list(d) + [0] * (max(n,p) - n)
        d_aug = np.array(d_aug) ** 2

        D = np.zeros((n,p))
        np.fill_diagonal(D, d)

        # print('d')
        # print(d)
        beta = beta_sampler()
        # print("beta")
        # print(beta)

        DtD_invs = {}

        # print(np.sum( d ** -2 ))
        # print(np.sum(d ** 2))
        # print(beta)

        # print("iterating over n_iters")
        # so it turns out this has serious problems with high dimensions
        # who would've thought 🙄
        # so actually we should be using the analytical expressions for the bias and variance you dumbass

        for k, reg in enumerate(regs):
            if k not in estimator_losses[x_iter].keys():
                estimator_losses[x_iter][k] = []

            if reg == 0:
                d_aug_copy = d_aug.copy()
                d_aug_copy[d_aug != 0] = d_aug[d_aug != 0] ** -1
                DtD_inv = np.diag(d_aug_copy)
            else:
                DtD_inv = np.diag((d_aug + reg) ** -1)
            
            # log asymptotic bias, exact bias, exact variance, total asy risk, total exact risk
            if reg != 0:
                tracked_theoreticals[x_iter][k] = [
                    fast_ridge_exact_bias(DtD_inv, D, O, beta), # regularization baked into DtD_inv
                    # alt_ridge_exp_bias(DtD_inv, D, O, beta),
                    ridge_exp_bias(X, d, beta, reg),
                    ridge_variance(X, sigma, d, reg),
                ]
            else:
                tracked_theoreticals[x_iter][k] = [
                    fast_ridgeless_exact_bias(DtD_inv, D, O, beta),
                    ridgeless_exp_bias(X, r, d,),
                    ridgeless_variance(X, sigma, d),
                ]

            DtD_invs[reg] = DtD_inv

        for _ in range(n_iters):
            
            y = X @ beta + noise_sampler()
            X_new, _, _, _, = x_sampler(return_svd = True)

            for k, reg in enumerate(regs):
                # (Xt X + r I)^{-1} Xt y = (Ot Dt D O + rI)^{-1} Ot Dt Q y
                # = Ot (DtD + rI)^{-1} Dt Q y
                beta_hat = O.T @ DtD_invs[reg] @ D.T @ Qt.T @ y
                estimator_losses[x_iter][k].append(
                    loss_func(X_new @ beta_hat, X_new @ beta)
                )


    return estimator_losses, tracked_theoreticals

def estimate_risk(loss_func, beta_hat, beta, x_sampler, n_iters):
    # print(loss_func)
    # print(beta_hat)
    # print(beta)
    out = []
    for _ in range(n_iters):
        X_new, _ = x_sampler()
        y_new = X_new @ beta # note there is no sigma
        y_pred = X_new @ beta_hat
 
        # print(y_new)
        # print(y_pred)
        out.append(
            loss_func(y_new, y_pred)
        )
    return out

def estimate_risk(loss_func, beta_hat, beta, x_sampler, n_iters):
    # print(loss_func)
    # print(beta_hat)
    # print(beta)
    out = []
    for _ in range(n_iters):
        X_new, _ = x_sampler()
        y_new = X_new @ beta # note there is no sigma
        y_pred = X_new @ beta_hat
 
        # print(y_new)
        # print(y_pred)
        out.append(
            loss_func(y_new, y_pred)
        )
    return out

def gaussian_asymptotic_pred(gamma, r, sigma):
    if gamma < 1:
        return sigma ** 2 * gamma / (1 - gamma)
    else:
        return r ** 2 * (1 - 1/gamma) + sigma ** 2 * 1 / (gamma - 1)

def svd_ridge_soln(Qt, d, O, y, lam):
    D_term = np.zeros((O.shape[0], Qt.shape[0]))

    # d_inv = d
    # d_inv[d != 0] = 1 / d[d != 0]
    d_inv = np.zeros(d.shape)
    d_inv[d != 0] = d[d != 0] / (d[d != 0] ** 2 + lam)

    np.fill_diagonal(D_term, d_inv)
    return O.T @ D_term @ Qt.T @ y

def compute_gcv_num(y, Qt, d, O, lam, divide = False):
    n = Qt.shape[0]
    # compute smoother matrix
    inner_matrix_diag = d ** 2 / (d ** 2 + lam)

    inner_matrix = np.zeros(Qt.shape)
    np.fill_diagonal(inner_matrix, inner_matrix_diag)

    # S_lam = X @ np.linalg.pinv(X.T @ X + r * np.eye(p)) @ X.T
    # print(np.trace(S_lam) / n)
    # print(S_lam[0])
    S_lam = Qt @ inner_matrix @ Qt.T
    # print(S_lam)
    if not divide:
        return 1/Qt.shape[0] * y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y
    else:
        return 1/Qt.shape[0] * y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y / (1 - np.trace(S_lam) / n) ** 2

def return_all_sts(n, p, d):
    if n > p or len(d) != n:
        assert False, "might be broken for underparam"
    d_aug = list(d) + [0] * (max(n, p) - n)
    d_aug = np.array(d_aug)
    def st(lam):
        return 1/p * np.sum( (d_aug ** 2 - lam) ** -1 )
    def stp(lam):
        return 1/p * np.sum( (d_aug ** 2 - lam) ** -2 )
    def cst(lam):
        return 1/n * np.sum( (d ** 2 - lam) ** -1 )
    def cstp(lam):
        return 1/n * np.sum( (d ** 2 - lam) ** -2 )
    
    return st, stp, cst, cstp

def t1_form(beta, Qt, inv_D, O, D,):
    p = O.shape[0]
    return beta.T @ O.T @ (np.eye(p) - D.T @ D @ inv_D) @ D.T @ Qt.T

def t3_form(eps, Qt, inv_D, O, D):
    n = Qt.shape[0]
    return (np.eye(n) - Qt @ D @ inv_D @ D.T @ Qt.T) @ eps

def diagnostic_gcv(X, Qt, d, O, beta, r, sigma, lambda_range, n_iters = 20, b_iters = 1):
    n, p = Qt.shape[0], O.shape[0]
    all_diagnostics = {
        "lambda": [],
        "n_iter": [],
        "gcv_num": [],
        "T1": [],
        "ET1": [],
        "T2": [],
        "T3": [],
        "ET3": [],
        "r2_coef": [],
        "s2_coef": [],
    }
    # true_y = X @ beta

    inv_Ds = {}
    D = np.zeros((n, p))
    np.fill_diagonal(D, d)

    for l in lambda_range:
        inv_Ds[l] = np.zeros((p, p))
        np.fill_diagonal(inv_Ds[l], 1 / (d ** 2 + l))

    eps_array = [gaussian_noise(n, sigma) for _ in range(n_iters)]

    t3_array = [
        [t3_form(eps, Qt, inv_Ds[l], O, D) for l in lambda_range] for eps in eps_array
    ]

    t1_array = [
        t1_form(beta, Qt, inv_Ds[l], O, D) for l in lambda_range
    ]

    st, stp, cst, cstp = return_all_sts(n, p, d)

    for l_index, l in enumerate(lambda_range):
        for i_index, i in enumerate(range(n_iters)):
            t1 = t1_array[l_index]
            t3 = t3_array[i_index][l_index]
            total_residuals = t1.T @ t1 + 2 * t1.T @ t3 + t3.T @ t3

            all_diagnostics['lambda'].append(l)
            all_diagnostics['n_iter'].append(i)
            all_diagnostics['gcv_num'].append(total_residuals)
            all_diagnostics['T1'].append(np.linalg.norm(t1, 2) ** 2)
            all_diagnostics['T3'].append(np.linalg.norm(t3, 2) ** 2)
            all_diagnostics['T2'].append(2 * t1.T @ t3)
            all_diagnostics['ET1'].append(
                r ** 2 * l ** 2 / (p/n) * (cst(-l) - l * cstp(-l)) * n
            )
            all_diagnostics['ET3'].append(
                sigma ** 2 * l ** 2 * cstp(-l) * n
            )
            all_diagnostics['r2_coef'].append(
                l ** 2 / (p/n) * (cst(-l) - l * cstp(-l)) * n
            )
            all_diagnostics['s2_coef'].append(
                l ** 2 * cstp(-l) * n
            )
    return pd.DataFrame(all_diagnostics)

def full_r2_s2_est(y, Qt, d, O, lambda_range, n, p):
    print("update")
    # factor = np.sqrt(np.sum(d ** 2) / p)
    # print(np.sum(d ** 2) / factor ** 2 / p)
    # print(factor)

    # gcv_nums = [
    #     compute_gcv_num(y, Qt, d, O, l) for l in lambda_range
    # ]
    # r2, s2 = r2_s2_estimator(lambda_range, gcv_nums, n, p, d)
    # return r2, s2
    factor = np.sqrt(np.sum(d ** 2) / p)
    print(np.sum(d ** 2) / factor ** 2 / p)
    print(factor)
    gcv_nums = [
        compute_gcv_num(y, Qt, d / factor, O, l) for l in lambda_range
    ]
    r2, s2 = r2_s2_estimator(lambda_range, gcv_nums, n, p, d / factor)
    return r2 / factor ** 2, s2



def r2_s2_estimator(lambda_range, gcv_nums, n, p, d):
    # gcv num is the value that has been divided by n
    st, stp, cst, cstp = return_all_sts(n, p, d)
    r2_coefs = np.array([
        l ** 2 / (p/n) * (cst(-l) - l * cstp(-l)) for l in lambda_range
    ])

    s2_coefs = np.array([
        l ** 2 * cstp(-l) for l in lambda_range
    ])

    gcv_num_np = np.array(gcv_nums)

    # compute r2 coefficient
    scaled_gcv_r2 = gcv_num_np / s2_coefs
    scaled_r2_coefs = r2_coefs / s2_coefs

    scaled_gcv_r2 -= scaled_gcv_r2.min()
    scaled_r2_coefs -= scaled_r2_coefs.min()

    est_r2 = (scaled_gcv_r2.T @ scaled_r2_coefs) / (scaled_r2_coefs.T @ scaled_r2_coefs)

    scaled_gcv_s2 = gcv_num_np / r2_coefs
    scaled_s2_coefs = s2_coefs / r2_coefs

    scaled_gcv_s2 -= scaled_gcv_s2.min()
    scaled_s2_coefs -= scaled_s2_coefs.min()

    est_s2 = (scaled_gcv_s2.T @ scaled_s2_coefs) / (scaled_s2_coefs.T @ scaled_s2_coefs)

    return est_r2, est_s2

def compute_gcv_num(y, Qt, d, O, lam, divide = False):
    if Qt.shape[0] > O.shape[0]:
        assert False, "Broken for underparameterized"
    n = Qt.shape[0]
    # compute smoother matrix
    inner_matrix_diag = d ** 2 / (d ** 2 + lam)

    inner_matrix = np.zeros(Qt.shape)
    np.fill_diagonal(inner_matrix, inner_matrix_diag)

    # S_lam = X @ np.linalg.pinv(X.T @ X + r * np.eye(p)) @ X.T
    # print(np.trace(S_lam) / n)
    # print(S_lam[0])
    S_lam = Qt @ inner_matrix @ Qt.T
    # print(S_lam)
    if not divide:
        return 1/Qt.shape[0] * y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y
    else:
        return 1/Qt.shape[0] * y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y / (1 - np.trace(S_lam) / n) ** 2

def compute_loocv(y, Qt, d, O, lam,):
    n = Qt.shape[0]
    p = O.shape[0]
    # compute smoother matrix

    d = d.copy()
    d = list(d) + [0] * (n - min(n, p))
    d = np.array(d)

    inner_matrix_diag = d ** 2 / (d ** 2 + lam)

    inner_matrix = np.zeros(Qt.shape)
    np.fill_diagonal(inner_matrix, inner_matrix_diag)
    # print(inner_matrix)

    # S_lam = X @ np.linalg.pinv(X.T @ X + r * np.eye(p)) @ X.T
    # print(np.trace(S_lam) / n)
    # print(S_lam[0])
    S_lam = Qt @ inner_matrix @ Qt.T
    # print(S_lam)
    D_lambda = np.diag((1 - np.diag(S_lam)) ** -1)
    # print(D_lambda)

    # print((D_lambda @ (np.eye(n) - S_lam) @ y) ** 2)

    return 1/Qt.shape[0] * (y.T @ (np.eye(n) - S_lam) @ D_lambda) @ (D_lambda @ (np.eye(n) - S_lam) @ y)

    # # print(S_lam)
    # if not divide:
    #     return 1/Qt.shape[0] * y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y
    # else:
    #     return 1/Qt.shape[0] * y.T @ (np.eye(n) - S_lam) @ (np.eye(n) - S_lam) @ y / (1 - np.trace(S_lam) / n) ** 2

def compute_mod_gcv(r, sigma, Qt, d, O, lam):
    n = Qt.shape[0]
    p = O.shape[0]
    d_aug = list(d) + [0] * (max(n, p) - n)
    d_aug = np.array(d_aug)
    gamma = p/n
    # alpha = r ** 2 / sigma ** 2 / gamma
    st, stp, cst, cstp = return_all_sts(n, p, d)

    print(r ** 2 * (lam ** 2 / gamma * cstp(-lam) + (gamma - 1) / gamma))
    print(sigma ** 2 * (cst(-lam) - lam * cstp(-lam)) )

    return (
        r ** 2 * (lam ** 2 / gamma * cstp(-lam) + (gamma - 1) / gamma) + 
        sigma ** 2 * (cst(-lam) - lam * cstp(-lam)) 
        )

def gaussian_asy_ridge_risk(lambda_range, gamma, r, sigma):
    # ok fuck this, just gonna use empirical distribution
    _, d = gaussian_sample(1000, 2000, )

    st, stp, cst, cstp = return_all_sts(1000, 2000, d)

    alpha = r ** 2 / (sigma ** 2 * gamma)
    preds = [
        sigma ** 2 * gamma * (
            st(-l) - l * (1 - alpha * l) * stp(-l)
        )
        for l in lambda_range
    ]    
    return preds

def semisynth(range_n, p, p_plus, x_train_pool, x_test_pool, r = 1, sigma = 1, x_iters = 100, unit_beta = None, multi_starts = None, seed = None):
    if seed is not None:
        np.random.seed(seed)
    all_mses = {}
    theos = {}
    if unit_beta is None:
        beta = beta_sampler(p, r = 1)
        print(beta[0])
    else:
        beta = unit_beta / np.linalg.norm(unit_beta) * r

    for curr_n in range_n:
        # data = speech_data['data'] / speech_data['data'].std()
        # data -= data.mean(0)
        # data = data / np.sqrt(n)
        # x_train = data[:n].to_numpy()
        # x_test = data[n:].to_numpy()
        # x_train /= x_train.std(0)
        # x_test /= x_test.std(0)
        # data = speech_data['data'] / speech_data['data'].std()
        # data -= data.mean(0)

        # data = data / np.sqrt(curr_n)
        if multi_starts is None:
            multi_starts = [0]
        beta *= np.sqrt(curr_n) / np.linalg.norm(beta) * r

        for m in multi_starts:
            if len(x_train_pool) - m < curr_n:
                continue


            x_train = x_train_pool[m:(m + curr_n)] / np.sqrt(curr_n)
            x_test = x_test_pool[m:(m + curr_n)] / np.sqrt(curr_n)
            # x_test = x_test_pool / np.sqrt(curr_n)
            Qt, d, O = np.linalg.svd(x_train)
            if curr_n > p_plus:
                d[p_plus:] = 0

            mses = []
            # beta = final_vector * np.sqrt(curr_n) / np.linalg.norm(final_vector) * r

            for i in range(x_iters):
                # print(x_train.shape)
                # print(beta.shape)
                # print(x_train.shape)
                # print(curr_n)
                eps = np.random.normal(size = curr_n, scale = sigma)
                # print(curr_n, m, i, eps[0])
                train_target = x_train @ beta + eps
                test_target = x_test @  beta # + np.random.normal(size = x_test.shape[0], scale = sigma)
                # y_train = target[:curr_n] + np.random.normal(size = curr_n, scale = sigma)
                beta_curr = svd_ridge_soln(Qt, d, O, train_target, 0)
                mse = np.mean(
                    (test_target - x_test @ beta_curr) ** 2
                )
                mses.append(mse)
            all_mses[(curr_n, m)] = np.array(mses)
            theos[(curr_n, m)] = (ridgeless_exp_bias(x_train, r, d) + ridgeless_variance(x_train, sigma, d)) * np.trace(x_test.T @ x_test) / x_test.shape[0] / p * curr_n
    return theos, all_mses

def test_sp_regression(range_n, p, p_plus, x_train_pool, x_test_pool, r = 1, sigma = 1, x_iters = 100, multi_starts = None, seed = None):
    if seed is not None:
        np.random.seed(seed)

    all_mses = {}
    theos = {}
    # beta = beta_sampler(p, r = 1)
    # print(beta[0])
    print("using real data to eval")
    for curr_n in range_n:
        # print(curr_n)
        # data = speech_data['data'] / speech_data['data'].std()
        # data -= data.mean(0)
        # data = data / np.sqrt(n)
        # x_train = data[:n].to_numpy()
        # x_test = data[n:].to_numpy()
        # x_train /= x_train.std(0)
        # x_test /= x_test.std(0)
        # data = speech_data['data'] / speech_data['data'].std()
        # data -= data.mean(0)
        
        if multi_starts is None:
            multi_starts = [0]
        
        for m in multi_starts:
            if len(x_train_pool) - m < curr_n:
                continue

            # data = data / np.sqrt(curr_n)
            x_train = x_train_pool[m:(m + curr_n)].drop(['SPY', 'SPY_proxy'], axis = 1) / np.sqrt(curr_n)
            # print(x_train.isna().any().any())
            # print(np.isinf(x_train).any().any())
            x_test = x_test_pool[m:(m + curr_n)].drop(['SPY', 'SPY_proxy'], axis = 1) / np.sqrt(curr_n)

            Qt, d, O = np.linalg.svd(x_train)
            if curr_n > p_plus:
                d[p_plus:] = 0

            # beta *= r / np.linalg.norm(beta) * np.sqrt(curr_n)

            eps = gaussian_noise(curr_n, sigma)
            # print(curr_n, m, eps[0])
            # train_target = x_train @ beta + eps
            # test_target = x_test @ beta # + gaussian_noise(x_test.shape[0], sigma)
            train_target = x_train_pool[m:(m + curr_n)]['SPY'] 
            test_target = x_test_pool[m:(m + curr_n)]['SPY']

            # mses = []

            beta_curr = svd_ridge_soln(Qt, d, O, train_target, 0)        
            mse = np.mean(
                (test_target - x_test @ beta_curr) ** 2
            )
            all_mses[(curr_n, m)] = mse

            # for i in range(x_iters):
            #     # print(x_train.shape)
            #     # print(beta.shape)
            #     train_target = x_train @ beta + np.random.normal(size = curr_n, scale = sigma)
            #     test_target = x_test @  beta # + np.random.normal(size = target[n:].shape, scale = sigma)
            #     # y_train = target[:curr_n] + np.random.normal(size = curr_n, scale = sigma)
            #     beta_curr = h.svd_ridge_soln(Qt, d, O, train_target, 0)
            #     mse = np.mean(
            #         (test_target - x_test @ beta_curr) ** 2
            #     )
            #     mses.append(mse)
            # all_mses[curr_n] = np.array(mses)
            theos[(curr_n, m)] = (ridgeless_exp_bias(x_train, r, d) + ridgeless_variance(x_train, sigma, d)) * np.trace(x_test.T @ x_test) / x_test.shape[0] / p_plus * curr_n
    return theos, all_mses

import math
def angle_between_vectors(vector1, vector2):
    # Normalize the vectors
    norm_vector1 = vector1 / np.linalg.norm(vector1)
    norm_vector2 = vector2 / np.linalg.norm(vector2)
    
    # Calculate the dot product
    dot_product = np.dot(norm_vector1, norm_vector2)
    
    # Calculate the angle using arccos
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Convert the angle from radians to degrees
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def angles_between_vec_matrix(vector1, vector2_matrix):
    # Normalize the vectors
    norm_vector1 = vector1 / np.linalg.norm(vector1)
    norm_vector2_matrix = vector2_matrix / np.linalg.norm(vector2_matrix, axis=1, keepdims=True)
    
    # Calculate the dot products
    dot_products = np.dot(norm_vector2_matrix, norm_vector1)
    
    # Calculate the angles using arccos
    angles_rad = np.arccos(np.clip(dot_products, -1.0, 1.0))
    
    # Convert the angles from radians to degrees
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg



def compute_tuned_risk(all_risks, all_est_gcv, all_bad_gcv):
    all_risks = np.array(all_risks)
    all_bad_gcv = np.array(all_bad_gcv)
    all_est_gcv = np.array(all_est_gcv)

    est_gcv_risks = np.array(all_risks)[np.arange(20),np.array(all_est_gcv).argmin(axis = 1) ]
    bad_gcv_risks = np.array(all_risks)[np.arange(20),np.array(all_bad_gcv).argmin(axis = 1) ]
    min_all_risks = np.min(all_risks, axis = 1)

    return min_all_risks, est_gcv_risks, bad_gcv_risks,

def compute_all_for_X_real(X_train, X_test, beta, lambdas, r, sigma, n_iters = 20):
    Qt, d, O = np.linalg.svd(X_train)
    n, p = X_train.shape

    true_y = X_test @ beta
    D = np.zeros((n,p))
    np.fill_diagonal(D, d)
    # 3 things to compute: our risk, which is averaged over eps??? the real answer is that they should be about the same...
    DtD_invs = {}
    d_aug = list(d) + [0] * (max(n,p) - n)
    d_aug = np.array(d_aug) ** 2

    for reg in lambdas:
        if reg == 0:
            d_aug_copy = d_aug.copy()
            d_aug_copy[d_aug != 0] = d_aug[d_aug != 0] ** -1
            DtD_inv = np.diag(d_aug_copy)
        else:
            DtD_inv = np.diag((d_aug + reg) ** -1)
        DtD_invs[reg] = DtD_inv

    # [print(DtD_invs[l].shape) for l in lambdas]
    theoretical_risks = [
        fast_ridge_exact_bias(DtD_invs[l], D, O, beta) + ridge_variance(X_train, sigma, d, l) for l in lambdas
        # h.ridge_exp_bias(X, d, beta, l) + h.ridge_variance(X, sigma, d, l) for l in lambdas
    ]

    all_actual_risks = []
    all_bad_gcv_risks = []
    all_oracle_gcv_risks = []
    all_est_gcv_risks = []
    all_r2s, all_s2s = [], []

    for i in range(n_iters):
        bad_gcv_risks = []
        oracle_gcv_risks = []
        est_gcv_risks = []
        eps = np.random.normal(size = (n,), scale = sigma)
        y = X_train @ beta + eps

        actual_risks = [
            np.linalg.norm(
                # h.svd_ridge_soln(Qt, d, O, y, l) - beta
                true_y - X_test @ svd_ridge_soln(Qt, d, O, y, l)
            ) ** 2 / X_test.shape[0] for l in lambdas
        ]

        r2_est, s2_est = full_r2_s2_est(y, Qt, d, O, np.logspace(0, 2, 10), n, p)
        all_r2s.append(r2_est)
        all_s2s.append(s2_est)
        print(r2_est, s2_est)
        # r2_est = 1 + np.random.normal(scale = 0.15)
        for lam in lambdas:
    
            # print(sigma_est, r2_est)
            bad_gcv_risks.append(compute_gcv_num(y, Qt, d, O, lam, divide = True) - sigma ** 2)
            oracle_gcv_risks.append(compute_mod_gcv(r, sigma, Qt, d, O, lam))
            est_gcv_risks.append(compute_mod_gcv(
                # r2_est,normal
                r2_est ** 0.5,
                s2_est ** 0.5,
                Qt, d, O, lam
            ))

        all_actual_risks.append(actual_risks)
        all_bad_gcv_risks.append(bad_gcv_risks)
        all_oracle_gcv_risks.append(oracle_gcv_risks)
        all_est_gcv_risks.append(est_gcv_risks)

    return theoretical_risks, all_actual_risks, all_bad_gcv_risks, all_oracle_gcv_risks, all_est_gcv_risks, all_r2s, all_s2s    

def compute_all_for_X(X, beta, lambdas, r, sigma, n_iters = 20):
    Qt, d, O = np.linalg.svd(X)
    n, p = X.shape
    D = np.zeros((n,p))
    np.fill_diagonal(D, d)
    # 3 things to compute: our risk, which is averaged over eps??? the real answer is that they should be about the same...
    DtD_invs = {}
    d_aug = list(d) + [0] * (max(n,p) - n)
    d_aug = np.array(d_aug) ** 2

    for reg in lambdas:
        if reg == 0:
            d_aug_copy = d_aug.copy()
            d_aug_copy[d_aug != 0] = d_aug[d_aug != 0] ** -1
            DtD_inv = np.diag(d_aug_copy)
        else:
            DtD_inv = np.diag((d_aug + reg) ** -1)
        DtD_invs[reg] = DtD_inv

    # [print(DtD_invs[l].shape) for l in lambdas]
    theoretical_risks = [
        fast_ridge_exact_bias(DtD_invs[l], D, O, beta) + ridge_variance(X, sigma, d, l) for l in lambdas
        # ridge_exp_bias(X, d, beta, l) + ridge_variance(X, sigma, d, l) for l in lambdas
    ]

    all_actual_risks = []
    all_bad_gcv_risks = []
    all_oracle_gcv_risks = []
    all_est_gcv_risks = []
    all_r2s, all_s2s = [], []

    for i in range(n_iters):
        bad_gcv_risks = []
        oracle_gcv_risks = []
        est_gcv_risks = []
        eps = np.random.normal(size = (n,), scale = sigma)
        y = X @ beta + eps

        actual_risks = [
            np.linalg.norm(
                svd_ridge_soln(Qt, d, O, y, l) - beta
            ) ** 2 / n for l in lambdas
        ]

        r2_est, s2_est = full_r2_s2_est(y, Qt, d, O, np.logspace(0, 2, 10), n, p)
        all_r2s.append(r2_est)
        all_s2s.append(s2_est)
        print(r2_est, s2_est)
        # r2_est = 1 + np.random.normal(scale = 0.15)
        for lam in lambdas:
    
            # print(sigma_est, r2_est)
            bad_gcv_risks.append(compute_gcv_num(y, Qt, d, O, lam, divide = True) - sigma ** 2)
            oracle_gcv_risks.append(compute_mod_gcv(r, sigma, Qt, d, O, lam))
            est_gcv_risks.append(compute_mod_gcv(
                # r2_est,normal
                r2_est ** 0.5,
                s2_est ** 0.5,
                Qt, d, O, lam
            ))

        all_actual_risks.append(actual_risks)
        all_bad_gcv_risks.append(bad_gcv_risks)
        all_oracle_gcv_risks.append(oracle_gcv_risks)
        all_est_gcv_risks.append(est_gcv_risks)

    return theoretical_risks, all_actual_risks, all_bad_gcv_risks, all_oracle_gcv_risks, all_est_gcv_risks, all_r2s, all_s2s

def compute_tuned_risk(all_risks, all_est_gcv, all_bad_gcv):
    all_risks = np.array(all_risks)
    all_bad_gcv = np.array(all_bad_gcv)
    all_est_gcv = np.array(all_est_gcv)

    n_iters = len(all_risks)

    est_gcv_risks = np.array(all_risks)[np.arange(n_iters),np.array(all_est_gcv).argmin(axis = 1) ]
    bad_gcv_risks = np.array(all_risks)[np.arange(n_iters),np.array(all_bad_gcv).argmin(axis = 1) ]
    min_all_risks = np.min(all_risks, axis = 1)

    return min_all_risks, est_gcv_risks, bad_gcv_risks,

def cond_cov(p, k, d_new, O):

    d_tail = sum(d_new[k:] ** 2) / (p - k)

    spike_cov = np.zeros((p, p))
    tail_cov = np.zeros((p, p))
    for i in range(k):
        spike_cov += (d_new[i] ** 2 - d_tail) * np.outer(O[i], O[i])
    tail_cov = d_tail * np.eye(p)

    return spike_cov + tail_cov, spike_cov, tail_cov

def alignment_bias(n, p, d, d_new, alphas, lam, r2):
    k = len(alphas)

    assert len(d) == min(n, p)
    assert len(d_new) <= p

    # r2 is norm of unaligned portion
    d_new2 = d_new ** 2
    d_tail = sum(d_new2[k:]) / (p - k)
    # print(d_tail)
    # print("spectrum tail", d_tail)
    d_aug = np.array(list(d) + (max(n, p) - n) * [0])
    alphas = np.zeros_like(alphas)
    
    unaligned_bias = (d_tail * lam ** 2 * np.sum((d_aug ** 2 + lam) ** (-2)) / p + np.sum(
        lam ** 2 * (d_aug[:k] **2 + lam) ** (-2) * (d_new2[:k] - d_tail)
    ) / p) * n * r2 # n is norm of unaligned beta

    aligned_bias = sum(
        [
            alphas[i] ** 2 * d_new[i] ** 2 * lam ** 2 * 1 / (d[i] ** 2 + lam) ** 2 for i in range(k)
        ]
    )
    
    return unaligned_bias + aligned_bias, unaligned_bias, aligned_bias

def alignment_var(n, p, d, d_new, k, lam, s2):
    d_new2 = d_new ** 2
    d_tail = sum(d_new2[k:]) / (p - k)
    # d_tail = 1

    # var_term = 0

    d2 = d ** 2
    spike_part = d2[:k] / (d2[:k] + lam) ** 2 * (d_new2[:k] - d_tail)
    spike_part = spike_part[:k] * s2
    # spike_part

    tail_part = d_tail * (d2 / (d2 + lam) ** 2) * s2

    return np.sum(spike_part) + np.sum(tail_part), np.sum(tail_part), np.sum(spike_part) # is correct finally

def evaluate(model, criterion, test_inputs, test_targets):
    with torch.no_grad():
        # Forward pass on test data
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        
    return test_loss.item()

import torch
import torch.nn as nn

def train(model, criterion, optimizer, train_inputs, train_targets, test_inputs, test_targets, num_epochs):

    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(train_inputs)
        loss = criterion(outputs, train_targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the loss for every epoch
        # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}")
        train_losses.append(loss.item())
        
        # Evaluate on the test set
        test_loss = evaluate(model, criterion, test_inputs, test_targets)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}")
        test_losses.append(test_loss)

    print(loss.item(), test_loss)
    return train_losses, test_losses

def oracle_risk(n, p, d, d_new, alphas, r2, s2, lam) :

    k = len(alphas)

    biases = alignment_bias(n, p, d, d_new, alphas, lam, r2)
    vars = alignment_var(n, p, d, d_new, k, lam, s2)

    # print(biases[0])
    # print(vars[0])
    return (biases[0] + vars[0])


def gcv_test_pipeline(X_train, y_train, X_test, y_test, lambdas, r, sigma, k, true_beta, n_iters = None, oracle_test = True):
    n, p = X_train.shape


    all_actual_risks = []
    all_new_gcv_risks = []
    all_bad_gcv_risks = []
    # all_oracle_gcv_risks = []
    all_loocv_risks = []

    print("Computing SVDs")
    Qt_train, d_train, O_train = np.linalg.svd(X_train)
    # print(O_train[0])
    Qt_test, d_test, O_test = np.linalg.svd(X_test)
    print("Done computing SVDs")

    y_true = X_train @ true_beta
    # print(y_true.shape)

    if not (y_train is None):
        assert n_iters is None or n_iters == 1

    for i in range(n_iters):
        eps = np.random.normal(size = n) * sigma
        # print("eps", eps[0])

        if y_train is None or i >=1: # start refreshing if more than one iteration
            y_train = y_true + eps
    
        # print(y_train[0], y_true[0], eps[0])
        bad_gcv_risks = [
            compute_gcv_num(y_train, Qt_train, d_train, O_train, lam = l, divide = True) for l in lambdas
        ]

        loocv_risks = [
            compute_loocv(y_train, Qt_train, d_train, O_train, lam = l,) for l in lambdas
        ]

        actual_risks = []
        for l in lambdas:
            beta_hat = svd_ridge_soln(Qt_train, d_train, O_train, y_train, l)
            # be careful here!!
            # beta_diff = true_beta - beta_hat
            # actual_risks.append(
            #     beta_diff.T @ pred_cov @ beta_diff
            # )
            if not oracle_test:
                actual_risks.append(
                    np.linalg.norm(y_test - X_test @ beta_hat) ** 2 / X_test.shape[0]
                )
            else:
                actual_risks.append(
                    np.linalg.norm(true_beta - beta_hat) ** 2 / n
                )
        
        X_pcr = (X_train @ O_train.T)[:, :k] # projection 
        pcr_coefs = np.linalg.pinv(X_pcr.T @ X_pcr) @ X_pcr.T @ y_train
        # # print(f"PCR Coefs: {pcr_coefs}")

        X_resid = (X_train @ O_train[k:n].T)
        beta_resid = np.linalg.pinv(X_resid.T @ X_resid) @ X_resid.T @ y_train
        D_comp = np.zeros((n - k, n - k))
        # if n <= p this works
        D_comp[np.arange(n - k), np.arange(n - k)] = d_train[k:]
        y_new = D_comp @ beta_resid
        X_new = D_comp @ O_train[k:n]
        
        # can speedup by not doing this
        # Qt_reduced, d_reduced, O_reduced = np.linalg.svd(X_new)

        O_permuted = O_train[list(range(k, n)) + list(range(0, k)) + list(range(n, p))]

        # print("d_train", d_train[k:10])

        r2, s2 = full_r2_s2_est(
            # y_train, Qt_train, d_train, O_train, np.logspace(0, 2, 10), n = n, p = p,
            y_new, np.eye(n - k), d_train[k:], O_permuted, np.logspace(0, 2.5, 6), n = n - k, p = p
        )

        # y_aligned = X_pcr @ pcr_coefs
        # y_resid = y_train - y_aligned


        # # X_resid = X_train - X_pcr @ O_train[:k]
        # d_train_mod = d_train.copy()
        # d_train_mod[:k] = 0

        # # print(y_aligned[0], y_resid[0], y_train[0])
        # r2, s2 = full_r2_s2_est(
        #     y_resid, Qt_train, d_train_mod, O_train, np.logspace(0, 2.5, 6), n, p
        # )
        print(r2, s2)

        new_gcv_risks = [
            # compute_mod_gcv(r2 ** 0.5, s2 ** 0.5, )
            oracle_risk(n, p, d_train, d_test, pcr_coefs, r2, s2, l) / X_test.shape[0] for l in lambdas
        ]
        # no such thing as an oracle value usually?
        # all_oracle_gcv_risks = [
        #     oracle_risk(n, p, d_train, d_test, alphas, r ** 2, sigma ** 2, l) for l in lambdas
        # ]

        # new_gcv_risks = [
        #     0 for l in lambdas
        # ]

        all_actual_risks.append(actual_risks)
        all_bad_gcv_risks.append(bad_gcv_risks)
        all_new_gcv_risks.append(new_gcv_risks)
        all_loocv_risks.append(loocv_risks)

    return all_actual_risks, all_bad_gcv_risks, all_loocv_risks, all_new_gcv_risks

def conditional_haar_draw(O, k):
    # return matrix with first k rows equal to O[:k, :]
    p = O.shape[0]

    O_tok = O[:k]
    O_fromk = O[k:]
    P_tok = np.eye(p)[:k]
    P_fromk = np.eye(p)[k:]

    tilde_O_new = ortho_group(
        dim = p - k,
    ).rvs(1)

    O_new = P_tok.T @ O_tok + P_fromk.T @ tilde_O_new @ O_fromk

    return O_new

import matplotlib.pyplot as plt

def plot_results(result_array, lambdas, s, filename = None, include_optimized_losses = True, ylim = None, square=True):
    # fig, ax = plt.subplots(1, 4, sharex = True, sharey = True, figsize = (11 / 1.3, 2.2 / 1.2)) 
    if square:
        fig, ax_grid = plt.subplots(2, 2, sharex = False, sharey = True, figsize = (7 / 2, 2.7 * 2)) 
        ax = ax_grid.flat
    else:
        fig, ax = plt.subplots(1, 4, sharex = True, sharey = True, figsize = (7, 2.7)) 
    # fig.tight_layout()

    colors = ['blue', 'green', 'red', 'purple']

    titles = ['Test Error', 'GCV', 'LOOCV', 'ROTI-GCV']

    count = len(result_array[0])

    for i in range(4):
        # plot each one alongside its mean
        mean = np.array(result_array[i]).mean(0)
        if i == 1 or i == 2:
            subtract_noise = 1
        else:
            subtract_noise = 0

        diffs_to_optimal = []
        chosen_risk = []

        for j in range(count):
            ax[i].plot(lambdas, np.array(result_array[i][j]) - (s ** 2) * subtract_noise, alpha = 0.3, color = colors[i])

            diffs_to_optimal.append(
                result_array[0][j][
                    np.argmin(result_array[i][j])
                ]# - np.min(result_array[0][j])
            )
            chosen_risk.append(
                np.min(result_array[i][j])
            )
        ax[i].plot(lambdas, mean - (s ** 2) * subtract_noise, alpha = 1, color = 'black')

        # compute diff to optimal

        if i == 0:
            # ax[i].set_title(f"{titles[i]}\nMinRisk: ${np.mean(chosen_risk):0.2f}\pm{np.std(chosen_risk) / np.sqrt(count):0.3f}$")
            ax[i].set_title(f"{titles[i]}\nMR: ${np.mean(chosen_risk):0.2f}$ $({np.std(chosen_risk) / np.sqrt(count):0.3f})$")
        else:
            # ax[i].set_title(f"{titles[i]}\nTunedRisk: ${np.mean(diffs_to_optimal):0.2f} \pm {np.std(diffs_to_optimal) / np.sqrt(count):0.3f}$\nEstRisk: ${np.mean(chosen_risk) - subtract_noise:0.2f} \pm {np.std(chosen_risk) / np.sqrt(count):0.3f}$")
            ax[i].set_title(f"{titles[i]}\nTR: ${np.mean(diffs_to_optimal):0.2f}$ $({np.std(diffs_to_optimal) / np.sqrt(count):0.3f})$\nER: ${np.mean(chosen_risk) - subtract_noise:0.2f}$ $({np.std(chosen_risk) / np.sqrt(count):0.3f})$")
            ax[i].plot(lambdas,
                       np.array(result_array[0]).mean(0),
                       '--')
        ax[i].set_xscale("log")
        ax[i].set_xlabel("$\lambda$", labelpad = -0)

    # ax[0].set_title("Actual MSE")
    # ax[1].set_title("GCV")
    # ax[2].set_title("LOOCV")
    # ax[3].set_title("newGCV")

    if ylim is not None:
        ax[3].set_ylim(ylim)

    # ax[4].set_title("One Run")
    # ax[4].set_xscale("log")
    # ax[4].set_xlabel("$\lambda$")
    # for i in range(4):
    #     if i == 1 or i == 2:
    #         subtract_noise = 1
    #     else:
    #         subtract_noise = 0

    #     ax[4].plot(lambdas, np.array(result_array[i][0]) - (s ** 2) * subtract_noise, color = colors[i])

    # print("hi")
    if square:
        fig.subplots_adjust(hspace=0, wspace=0)
    else:
        fig.subplots_adjust(hspace=0.0, wspace=0.1)
    fig.tight_layout()

    if filename is not None:
         fig.savefig(fname = filename, dpi = 600, bbox_inches = "tight")


def plot_results_combo(result_array_array, lambdas, s, filename = None, include_optimized_losses = True, ylim = None):
    fig, ax_arr = plt.subplots(len(result_array_array), 4, sharex = True, sharey = True, figsize = (11.5 / 1.6, 2.2 * 5.5 / 1.35 / 2)) 
    # fig, ax_grid = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (6, 6)) 
    # ax = ax_grid.flat

    colors = ['blue', 'green', 'red', 'purple']

    titles = ['Test Error', 'GCV', 'LOOCV', 'ROTI-GCV']

    count = len(result_array_array[0][0])

    for row_num in range(len(result_array_array)):
        for i in range(4):
            ax = ax_arr[row_num]
            result_array = result_array_array[row_num]
            # plot each one alongside its mean
            mean = np.array(result_array[i]).mean(0)
            if i == 1 or i == 2:
                subtract_noise = 1
            else:
                subtract_noise = 0

            diffs_to_optimal = []
            chosen_risk = []

            for j in range(count):
                ax[i].plot(lambdas, np.array(result_array[i][j]) - (s ** 2) * subtract_noise, alpha = 0.3, color = colors[i])

                diffs_to_optimal.append(
                    result_array[0][j][
                        np.argmin(result_array[i][j])
                    ]# - np.min(result_array[0][j])
                )
                chosen_risk.append(
                    np.min(result_array[i][j])
                )
            ax[i].plot(lambdas, mean - (s ** 2) * subtract_noise, alpha = 1, color = 'black')

            # compute diff to optimal

            if i == 0:
                # ax[i].set_title(f"{titles[i]}\nMinRisk: ${np.mean(chosen_risk):0.2f}\pm{np.std(chosen_risk) / np.sqrt(count):0.3f}$")
                ax[i].set_title(f"Trial {row_num + 3}\n{titles[i]}\nMR: ${np.mean(chosen_risk):0.2f}$ $({np.std(chosen_risk) / np.sqrt(count):0.3f})$")
            else:
                # ax[i].set_title(f"{titles[i]}\nTunedRisk: ${np.mean(diffs_to_optimal):0.2f} \pm {np.std(diffs_to_optimal) / np.sqrt(count):0.3f}$\nEstRisk: ${np.mean(chosen_risk) - subtract_noise:0.2f} \pm {np.std(chosen_risk) / np.sqrt(count):0.3f}$")
                ax[i].set_title(f"{titles[i]}\nTR: ${np.mean(diffs_to_optimal):0.2f}$ $({np.std(diffs_to_optimal) / np.sqrt(count):0.3f})$\nER: ${np.mean(chosen_risk) - subtract_noise:0.2f}$ $({np.std(chosen_risk) / np.sqrt(count):0.3f})$")
                ax[i].plot(lambdas,
                        np.array(result_array[0]).mean(0),
                        '--')
            ax[i].set_xscale("log")
            ax[i].set_xlabel("$\lambda$")

        # ax[0].set_title("Actual MSE")
        # ax[1].set_title("GCV")
        # ax[2].set_title("LOOCV")
        # ax[3].set_title("newGCV")

        if ylim is not None:
            ax[3].set_ylim(ylim)

    # ax[4].set_title("One Run")
    # ax[4].set_xscale("log")
    # ax[4].set_xlabel("$\lambda$")
    # for i in range(4):
    #     if i == 1 or i == 2:
    #         subtract_noise = 1
    #     else:
    #         subtract_noise = 0

    #     ax[4].plot(lambdas, np.array(result_array[i][0]) - (s ** 2) * subtract_noise, color = colors[i])

    fig.subplots_adjust(hspace=0.83,wspace=0.2)

    if filename is not None:
        fig.savefig(fname = filename, dpi = 600, bbox_inches = "tight")