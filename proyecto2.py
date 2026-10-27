# %% [markdown]
# # Proyecto 2: Contando clientes con modelos *Buy 'Til You Die*
#
# **Análisis y Ciencia de Datos — Universidad del Istmo de Guatemala**
#
# Este notebook corre de principio a fin a partir de los archivos originales
# (`data/CDNOW_sample.txt` y `data/Online Retail.xlsx`). El BG/NBD, el
# MBG/NBD y todas sus cantidades derivadas están programados a mano en el
# paquete `btyd/` usando solo `scipy.special.gammaln`, `betaln`, `hyp2f1` y
# `scipy.optimize.minimize`.

# %%
import inspect
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from btyd import bgnbd, data, mbgnbd

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.precision", 4)
pd.set_option("display.width", 120)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3})
FIG = Path("figuras")
FIG.mkdir(exist_ok=True)
RES = Path("resultados")
RES.mkdir(exist_ok=True)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / f"{nombre}.png", dpi=160, bbox_inches="tight")
    plt.show()


def tabla(df, nombre):
    df.to_csv(RES / f"{nombre}.csv")
    return df


# %% [markdown]
# ## 0. Los datos: preparación
#
# **0.1** Si un cliente compró dos veces el mismo día, se cuenta como una sola
# transacción (se suman CDs y montos). **0.2** El tiempo se mide en semanas
# (días/7) con el día 0 = 1 de enero de 1997. Calibración: 1-ene a 30-sep-1997
# (39 semanas); *holdout*: 1-oct-1997 a 30-jun-1998 (39 semanas).

# %%
trans = data.leer_cdnow()
crudo = pd.read_csv(data.DATA / "CDNOW_sample.txt", sep=r"\s+", header=None)
print(f"Registros en el archivo original: {len(crudo):,}")
print(f"Transacciones (cliente-día):       {len(trans):,}")
print(f"Clientes:                          {trans['id'].nunique():,}")
print("Rango de fechas:", trans["fecha"].min().date(), "a", trans["fecha"].max().date())

# %% [markdown]
# **0.3** Para cada cliente: $x$ = compras repetidas en calibración, $t_x$ =
# momento de la última compra medido desde la primera, $T$ = (30-sep − primera
# compra)/7 y $x^*$ = compras en el *holdout*.

# %%
d = data.cdnow_rfm()
x, tx, T, xs = (d[c].to_numpy(float) for c in ["x", "tx", "T", "x_star"])
print(d.head())
print(d[["x", "tx", "T", "x_star"]].describe().round(3))
print(f"\nCompras repetidas en calibración: {x.sum():.0f};  en holdout: {xs.sum():.0f}")

# %%
frac0 = (x == 0).mean()
print(f"0.4  Fracción sin compras repetidas en calibración: {frac0:.3f} "
      f"({(x == 0).sum()} de {len(x)})")
print(f"     T va de {T.min():.2f} a {T.max():.2f} semanas")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].hist(d["t0"], bins=np.arange(0, 13.5, 0.5), color="C0")
ax[0].set(title="Semana de la primera compra", xlabel="semana", ylabel="clientes")
ax[1].hist(T, bins=40, color="C1")
ax[1].set(title="T (edad al 30-sep)", xlabel="semanas")
cnt = pd.Series(np.minimum(x, 10)).value_counts().sort_index()
ax[2].bar(cnt.index, cnt.values, color="C2")
ax[2].set(title="x en calibración (10 = 10+)", xlabel="x")
guardar(fig, "00_datos")

# %% [markdown]
# **0.4 Respuesta.** El 59.9% de los clientes (1,411 de 2,357) no volvió a
# comprar durante las 39 semanas de calibración: el "grupo silencioso" es la
# mayoría, y cualquier modelo tiene que decir algo sensato sobre ellos.
#
# $T$ es distinto para cada cliente porque la cohorte se **adquirió de forma
# escalonada** a lo largo del primer trimestre (primeras compras del 1 de enero
# al 25 de marzo), pero la calibración termina en la misma fecha calendario
# para todos (30 de septiembre). Un cliente que llegó el 1 de enero tuvo 38.86
# semanas para recomprar; uno que llegó el 25 de marzo solo 27. Por eso la
# verosimilitud condiciona en $T_i$: no ver compras en una ventana corta dice
# menos que no verlas en una larga. (En la gráfica de la izquierda también se
# ve el patrón de adquisición: muchos clientes la primera semana de enero.)

# %% [markdown]
# ---
# ## Parte 1: el modelo, pieza por pieza
#
# Supuestos del BG/NBD: (i) mientras está activo, el cliente $i$ compra según
# un proceso de Poisson con tasa $\lambda_i$; (ii) $\lambda \sim
# \text{Gamma}(r,\alpha)$; (iii) después de **cada** compra (repetida) muere
# con probabilidad $p_i$; (iv) $p\sim\text{Beta}(a,b)$; (v) $\lambda_i \perp p_i$.
#
# ### 1.1 Un cliente
#
# Fijemos $\lambda$ y $p$. Observamos $x$ compras repetidas en $(0,T]$ en los
# momentos $t_1<\dots<t_x=t_x$, y ninguna en $(t_x, T]$. Hay **dos historias**
# que producen exactamente estos datos:
#
# **Historia A — sigue vivo en $T$.** Los tiempos entre compras son
# exponenciales: la densidad de las $x$ compras es
# $\lambda e^{-\lambda t_1}\cdot\lambda e^{-\lambda(t_2-t_1)}\cdots\lambda e^{-\lambda(t_x-t_{x-1})}
# = \lambda^x e^{-\lambda t_x}$. Sobrevivió a cada una de las $x$ compras:
# $(1-p)^x$. Y, estando vivo, no compró en $(t_x,T]$: $e^{-\lambda(T-t_x)}$.
# Multiplicando,
# $$L_A=(1-p)^x\,\lambda^x e^{-\lambda T}.$$
#
# **Historia B — murió justo después de la $x$-ésima compra.** La densidad de
# las compras es la misma, $\lambda^x e^{-\lambda t_x}$; sobrevivió a las
# primeras $x-1$, $(1-p)^{x-1}$, y murió después de la última, $p$. Ya muerto,
# no comprar en $(t_x, T]$ tiene probabilidad 1:
# $$L_B = p(1-p)^{x-1}\lambda^x e^{-\lambda t_x}.$$
#
# Las historias son excluyentes, así que la verosimilitud es la suma:
# $$L(\lambda,p\mid x,t_x,T) = (1-p)^x\lambda^x e^{-\lambda T} + \delta_{x>0}\,p(1-p)^{x-1}\lambda^x e^{-\lambda t_x}.$$
#
# **¿Por qué la historia B no existe si $x=0$?** En el BG/NBD la muerte solo
# puede ocurrir *justo después de una compra repetida*. Si $x=0$ no hubo
# ninguna compra en $(0,T]$ después de la cual morir; la primera compra (en
# $t=0$) no cuenta como oportunidad de muerte. Un cliente con $x=0$ está, por
# construcción, vivo en $T$. (Esta es la pieza que va a importar en la Parte 4.)
#
# Nótese que solo aparecen $x$, $t_x$ y $T$: los momentos de las compras
# intermedias se cancelan, por eso $(x,t_x,T)$ son estadísticos suficientes.
#
# ### 1.2 La mezcla
#
# Integramos sobre $\lambda\sim\text{Gamma}(r,\alpha)$, con densidad
# $g(\lambda)=\alpha^r\lambda^{r-1}e^{-\alpha\lambda}/\Gamma(r)$, y sobre
# $p\sim\text{Beta}(a,b)$, con densidad $f(p)=p^{a-1}(1-p)^{b-1}/B(a,b)$. Por
# (v), cada término se factoriza en una integral en $\lambda$ por una en $p$.
#
# **Integral en $\lambda$** (la misma de la mezcla gamma–Poisson): para $s\in\{T, t_x\}$,
# $$\int_0^\infty \lambda^x e^{-\lambda s}\frac{\alpha^r\lambda^{r-1}e^{-\alpha\lambda}}{\Gamma(r)}d\lambda
# =\frac{\alpha^r}{\Gamma(r)}\int_0^\infty\lambda^{r+x-1}e^{-(\alpha+s)\lambda}d\lambda
# =\frac{\Gamma(r+x)\,\alpha^r}{\Gamma(r)}\,\frac{1}{(\alpha+s)^{r+x}},$$
# porque el integrando es el núcleo de una $\text{Gamma}(r+x,\alpha+s)$.
#
# **Integrales en $p$** (definición de la función beta):
# $$\int_0^1 (1-p)^x\frac{p^{a-1}(1-p)^{b-1}}{B(a,b)}dp=\frac{B(a,b+x)}{B(a,b)},\qquad
# \int_0^1 p(1-p)^{x-1}\frac{p^{a-1}(1-p)^{b-1}}{B(a,b)}dp=\frac{B(a+1,b+x-1)}{B(a,b)}.$$
#
# Juntando:
# $$L(r,\alpha,a,b\mid x,t_x,T)=\frac{\Gamma(r+x)\,\alpha^r}{\Gamma(r)}\left[\frac{B(a,b+x)}{B(a,b)}\frac{1}{(\alpha+T)^{r+x}}
# +\delta_{x>0}\frac{B(a+1,b+x-1)}{B(a,b)}\frac{1}{(\alpha+t_x)^{r+x}}\right].$$
#
# La log-verosimilitud de la muestra es $\sum_i \log L_i$. Una consecuencia
# útil: para $x=0$, $B(a,b+0)/B(a,b)=1$ y $L_i=\left(\frac{\alpha}{\alpha+T_i}\right)^r$
# **no depende de $a$ ni de $b$**. Los clientes sin recompras no aportan
# ninguna información sobre la muerte.

# %% [markdown]
# La implementación (archivo `btyd/bgnbd.py`) trabaja en logaritmos, usa
# `np.logaddexp` para sumar las dos historias y evalúa el segundo término solo
# para $x>0$ (poniendo $-\infty$ en su logaritmo cuando $x=0$):

# %%
print(inspect.getsource(bgnbd.loglik_individual))

# %% [markdown]
# **Chequeo de la integral.** Antes de simular, verificamos numéricamente que
# la fórmula cerrada de 1.2 es la integral de la de 1.1: tomamos millones de
# $(\lambda, p)$ de la gamma y la beta y promediamos $L(\lambda,p\mid x,t_x,T)$
# (Monte Carlo) para varios clientes.

# %%
rng = np.random.default_rng(1)
par_prueba = np.array([0.25, 4.0, 0.8, 2.5])
n_mc = 4_000_000
lam = rng.gamma(par_prueba[0], 1 / par_prueba[1], n_mc)
pp = rng.beta(par_prueba[2], par_prueba[3], n_mc)
filas = []
for (xi, txi, Ti) in [(0, 0, 30), (1, 10, 35), (3, 20, 38), (7, 30, 39), (2, 2, 27)]:
    L1 = (1 - pp) ** xi * lam ** xi * np.exp(-lam * Ti)
    if xi > 0:
        L1 = L1 + pp * (1 - pp) ** (xi - 1) * lam ** xi * np.exp(-lam * txi)
    mc = L1.mean()
    cerrada = np.exp(bgnbd.loglik_individual(par_prueba, [xi], [txi], [Ti]))[0]
    filas.append({"x": xi, "tx": txi, "T": Ti, "Monte Carlo": mc,
                  "fórmula 1.2": cerrada, "error rel.": mc / cerrada - 1})
pd.DataFrame(filas)

# %% [markdown]
# ### 1.3 Primero, datos que nosotros inventamos
#
# Simulamos 2,357 clientes con $(r,\alpha,a,b) = (0.25, 4, 0.8, 2.5)$ y
# $T\sim U(27, 39)$ semanas, y estimamos con nuestra verosimilitud. Repetimos
# con **50 semillas**. La simulación sigue la historia del modelo: tiempos
# entre compras exponenciales con tasa $\lambda_i$ y, después de cada compra
# repetida, el cliente muere con probabilidad $p_i$.

# %%
VERDAD = np.array([0.25, 4.0, 0.8, 2.5])
N_SEM = 50
est = []
for s in range(N_SEM):
    rng = np.random.default_rng(1000 + s)
    Ts = rng.uniform(27, 39, 2357)
    xs_, txs_, Ts_, _, _ = bgnbd.simulate(VERDAD, Ts, rng)
    f = bgnbd.fit(xs_, txs_, Ts_, seed=s)
    est.append(np.r_[f["params"], f["loglik"], (xs_ == 0).mean()])
est = pd.DataFrame(est, columns=["r", "alpha", "a", "b", "loglik", "frac_x0"])
est["r/alpha"] = est["r"] / est["alpha"]
est["a/(a+b)"] = est["a"] / (est["a"] + est["b"])
verdad_ext = pd.Series({"r": 0.25, "alpha": 4.0, "a": 0.8, "b": 2.5,
                        "r/alpha": 0.25 / 4, "a/(a+b)": 0.8 / 3.3})
cols = list(verdad_ext.index)
res13 = pd.DataFrame({
    "verdadero": verdad_ext,
    "media": est[cols].mean(),
    "desv. est.": est[cols].std(),
    "sesgo %": 100 * (est[cols].mean() / verdad_ext - 1),
    "CV % (desv/verdad)": 100 * est[cols].std() / verdad_ext,
    "IC 95% cubre": [np.mean(np.abs(est[c] - verdad_ext[c]) <= 1.96 * est[c].std()) for c in cols],
})
tabla(res13.round(4), "1_3_simulacion")

# %%
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
for i, c in enumerate(["r", "alpha", "a", "b"]):
    ax[i].hist(est[c], bins=15, color=f"C{i}", alpha=0.8)
    ax[i].axvline(VERDAD[i], color="k", ls="--", label="verdadero")
    ax[i].axvline(est[c].mean(), color="red", label="media")
    ax[i].set_title(f"{c}: CV = {100 * est[c].std() / VERDAD[i]:.1f}%")
ax[0].legend()
guardar(fig, "1_3_simulacion")
print("Correlación entre estimados:")
print(est[["r", "alpha", "a", "b"]].corr().round(2))
print(f"\nFracción media de clientes con x=0 en las simulaciones: {est['frac_x0'].mean():.3f}")

# %% [markdown]
# **1.3 Respuesta.** Sí recuperamos los parámetros: las medias de los 50
# estimados quedan a menos de un error estándar del valor verdadero en los
# cuatro casos y los intervalos $\pm1.96$ desviaciones cubren el valor
# verdadero entre 92% y 96% de las veces. (El pequeño sesgo positivo de
# $a$ y $b$, de 7–10%, es el sesgo de muestra finita típico del MLE de una
# beta; es menos de medio error estándar.) Además, el chequeo de Monte Carlo
# de arriba confirma que la fórmula cerrada de 1.2 es la integral de 1.1.
#
# **¿Qué se estima con más precisión?** $r$ y $\alpha$ (CV de 4% y 7%) mucho
# mejor que $a$ y $b$ (CV de 24% y 29%). Razones:
#
# 1. **Información.** Todos los clientes informan sobre el proceso de compra
#    ($x$ y $T$ dicen cuánto compran), pero como mostramos en 1.2, los
#    clientes con $x=0$ (57% en la simulación) no aportan *nada* sobre $(a,b)$.
#    La muerte solo se aprende de los recompradores, y de forma indirecta: por
#    el silencio $T-t_x$ después de su última compra, que también podría
#    deberse a una $\lambda$ baja.
# 2. **Identificación débil de la forma de la beta.** $a$ y $b$ están muy
#    correlacionados entre simulaciones (0.93): la media $a/(a+b)$ se estima
#    bien (CV 7%), pero la *dispersión* de la beta (qué tan distintos son los
#    clientes en su $p$) se confunde fácilmente con cambiar $a$ y $b$ juntos.
#    Igual, $r$ y $\alpha$ están correlacionados (0.73) y su cociente
#    $r/\alpha$, la tasa media, tiene CV de 5%.
#
# Moraleja: los resultados que dependen del *promedio* (tasa media, $E[p]$,
# pronósticos agregados) son robustos; los que dependen de la *forma* de la
# heterogeneidad en la muerte merecen más cautela.

# %% [markdown]
