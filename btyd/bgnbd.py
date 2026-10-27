"""BG/NBD (Fader, Hardie y Lee, 2005) programado a mano.

Todo trabaja con arreglos de numpy: x (compras repetidas), tx (recencia) y T
(edad del cliente), en semanas. Los parámetros son (r, alpha, a, b).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

PARAMS = ("r", "alpha", "a", "b")


# ---------------------------------------------------------------------------
# Verosimilitud
# ---------------------------------------------------------------------------
def loglik_individual(params, x, tx, T):
    """log L_i(r, alpha, a, b | x_i, tx_i, T_i) para cada cliente (ec. 1.2)."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)

    # Gamma(r + x) alpha^r / Gamma(r)
    comun = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    # Historia 1: sigue vivo hasta T
    t1 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + T)
    # Historia 2: murió justo después de la x-ésima compra (solo si x > 0).
    # Se evalúa solo donde x > 0 para no meter betaln(a+1, b-1) basura.
    pos = x > 0
    t2 = np.full_like(x, -np.inf)
    xp = x[pos]
    t2[pos] = (betaln(a + 1, b + xp - 1) - betaln(a, b)
               - (r + xp) * np.log(alpha + tx[pos]))
    return comun + np.logaddexp(t1, t2)


def loglik(params, x, tx, T):
    return np.sum(loglik_individual(params, x, tx, T))


def _negll_log(logp, x, tx, T):
    val = -loglik(np.exp(logp), x, tx, T)
    return val if np.isfinite(val) else 1e300


def fit(x, tx, T, starts=None, bounds=(-10.0, 10.0), seed=0, n_random=4):
    """Máxima verosimilitud sobre log-parámetros.

    Corre L-BFGS-B (con cotas en los log-parámetros) desde varios puntos de
    arranque y pule el mejor con Nelder-Mead. Devuelve un dict con los
    parámetros, la log-verosimilitud y todos los arranques.
    """
    rng = np.random.default_rng(seed)
    if starts is None:
        starts = [np.zeros(4), np.log([0.5, 5.0, 1.0, 3.0])]
        starts += [rng.uniform(-2, 2, size=4) for _ in range(n_random)]
    runs = []
    for s0 in starts:
        res = minimize(_negll_log, s0, args=(x, tx, T), method="L-BFGS-B",
                       bounds=[bounds] * 4)
        runs.append(res)
    best = min(runs, key=lambda r: r.fun)
    nm = minimize(_negll_log, best.x, args=(x, tx, T), method="Nelder-Mead",
                  options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000,
                           "maxfev": 40000})
    if nm.fun < best.fun:
        best = nm
    p = np.exp(best.x)
    return {
        "params": p,
        "loglik": -best.fun,
        "logparams": best.x,
        "converged": bool(best.success),
        "runs": [(np.exp(r.x), -r.fun) for r in runs],
        "en_cota": bool(np.any(np.abs(best.x) > abs(bounds[1]) - 1e-3)),
    }


# ---------------------------------------------------------------------------
# Cantidades derivadas
# ---------------------------------------------------------------------------
def expected_x(t, params):
    """E[X(t)]: compras repetidas esperadas en (0, t] de un cliente nuevo."""
    r, alpha, a, b = params
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    z = t / (alpha + t)
    out = (a + b - 1) / (a - 1) * (
        1 - (alpha / (alpha + t)) ** r * hyp2f1(r, b, a + b - 1, z))
    return np.where(t > 0, out, 0.0)


def p_alive(params, x, tx, T):
    """P(vivo en T | x, tx, T)."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)
    # log del término a/(b+x-1) ((alpha+T)/(alpha+tx))^(r+x), solo x > 0
    pos = x > 0
    log_term = np.full_like(x, -np.inf)
    xp = x[pos]
    log_term[pos] = (np.log(a) - np.log(b + xp - 1)
                     + (r + xp) * (np.log(alpha + T[pos]) - np.log(alpha + tx[pos])))
    return np.exp(-np.logaddexp(0.0, log_term))


def conditional_expected(tau, params, x, tx, T):
    """E[Y(tau) | x, tx, T]: compras esperadas en (T, T + tau]."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    tx = np.asarray(tx, dtype=float)
    T = np.asarray(T, dtype=float)
    A = (alpha + T) / (alpha + T + tau)
    num = 1 - A ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, 1 - A)
    return (a + b + x - 1) / (a - 1) * num * p_alive(params, x, tx, T)


def p_buy(tau, params, x, tx, T):
    """P(Z = 1): probabilidad de al menos una compra en (T, T + tau] (inciso 4.3)."""
    r, alpha, a, b = params
    T = np.asarray(T, dtype=float)
    x = np.asarray(x, dtype=float)
    p0_vivo = ((alpha + T) / (alpha + T + tau)) ** (r + x)
    return p_alive(params, x, tx, T) * (1 - p0_vivo)


# ---------------------------------------------------------------------------
# Simulación
# ---------------------------------------------------------------------------
def simulate(params, T, rng, holdout=0.0):
    """Simula clientes BG/NBD con primera compra en 0 y calibración T_i.

    Devuelve x, tx, T, el número de compras en (T, T + holdout] y si el
    cliente seguía vivo en T. La muerte solo ocurre justo después de una
    compra repetida.
    """
    r, alpha, a, b = params
    T = np.asarray(T, dtype=float)
    n = len(T)
    lam = rng.gamma(shape=r, scale=1 / alpha, size=n)
    p = rng.beta(a, b, size=n)
    x = np.zeros(n)
    tx = np.zeros(n)
    xh = np.zeros(n)
    vivo_T = np.zeros(n, dtype=bool)
    fin = T + holdout
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i]) if lam[i] > 0 else np.inf
            if t > fin[i]:
                vivo_T[i] = True
                break
            if t <= T[i]:
                x[i] += 1
                tx[i] = t
            else:
                xh[i] += 1
            if rng.random() < p[i]:
                vivo_T[i] = t > T[i]
                break
    return x, tx, T, xh, vivo_T
