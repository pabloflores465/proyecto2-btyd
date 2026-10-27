"""MBG/NBD (Batislam, Denizel y Filiztekin, 2007).

Igual que el BG/NBD, pero el cliente también puede morir justo después de su
primera compra: con x compras repetidas hay x + 1 oportunidades de morir.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1


def loglik_individual(params, x, tx, T):
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)
    comun = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    t1 = betaln(a, b + x + 1) - betaln(a, b) - (r + x) * np.log(alpha + T)
    t2 = betaln(a + 1, b + x) - betaln(a, b) - (r + x) * np.log(alpha + tx)
    return comun + np.logaddexp(t1, t2)


def loglik(params, x, tx, T):
    return np.sum(loglik_individual(params, x, tx, T))


def _negll_log(logp, x, tx, T):
    val = -loglik(np.exp(logp), x, tx, T)
    return val if np.isfinite(val) else 1e300


def fit(x, tx, T, bounds=(-10.0, 10.0), seed=0, n_random=4):
    rng = np.random.default_rng(seed)
    starts = [np.zeros(4), np.log([0.5, 5.0, 1.0, 3.0])]
    starts += [rng.uniform(-2, 2, size=4) for _ in range(n_random)]
    runs = [minimize(_negll_log, s0, args=(x, tx, T), method="L-BFGS-B",
                     bounds=[bounds] * 4) for s0 in starts]
    best = min(runs, key=lambda r: r.fun)
    nm = minimize(_negll_log, best.x, args=(x, tx, T), method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000,
                           "maxfev": 40000})
    if nm.fun < best.fun:
        best = nm
    return {"params": np.exp(best.x), "loglik": -best.fun,
            "runs": [(np.exp(r.x), -r.fun) for r in runs]}


def p_alive(params, x, tx, T):
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)
    log_term = (np.log(a) - np.log(b + x)
                + (r + x) * (np.log(alpha + T) - np.log(alpha + tx)))
    return np.exp(-np.logaddexp(0.0, log_term))


def expected_x(t, params):
    r, alpha, a, b = params
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    out = b / (a - 1) * (1 - (alpha / (alpha + t)) ** r
                         * hyp2f1(r, b + 1, a + b, t / (alpha + t)))
    return np.where(t > 0, out, 0.0)


def conditional_expected(tau, params, x, tx, T):
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    T = np.asarray(T, dtype=float)
    A = (alpha + T) / (alpha + T + tau)
    num = 1 - A ** (r + x) * hyp2f1(r + x, b + x + 1, a + b + x, 1 - A)
    return (a + b + x) / (a - 1) * num * p_alive(params, x, tx, T)


def p_buy(tau, params, x, tx, T):
    r, alpha, a, b = params
    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)
    return p_alive(params, x, tx, T) * (1 - ((alpha + T) / (alpha + T + tau)) ** (r + x))
