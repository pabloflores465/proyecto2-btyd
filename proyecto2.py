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
# ---
# ## Parte 2: ajuste a CDNOW
#
# ### 2.1 Estimación

# %%
ajuste = bgnbd.fit(x, tx, T)
P = ajuste["params"]
r, alpha, a, b = P
art = np.array([0.243, 4.414, 0.793, 2.426])
tabla(pd.DataFrame({"nuestro": np.r_[P, ajuste["loglik"]],
                    "artículo": np.r_[art, -9582.4]},
                   index=["r", "alpha", "a", "b", "log-verosimilitud"]).round(4),
      "2_1_estimados")

# %% [markdown]
# Arranques múltiples: todos los arranques razonables llegan al mismo óptimo.
# Si alguno se queda en una cota (por ejemplo $a\approx 0$ o $b$ enorme) es un
# optimizador perdido, no un estimado, y su log-verosimilitud lo delata.

# %%
pd.DataFrame([np.r_[p, l] for p, l in ajuste["runs"]],
             columns=["r", "alpha", "a", "b", "loglik"]).round(4)

# %% [markdown]
# ### 2.2 Interpretación de los parámetros

# %%
media_lam = r / alpha
media_p = a / (a + b)
print(f"Tasa media de compra de un cliente activo  E[λ] = r/α = {media_lam:.4f} por semana "
      f"= {52 * media_lam:.2f} por año")
print(f"Desv. est. de λ = sqrt(r)/α = {np.sqrt(r) / alpha:.4f};  CV(λ) = 1/sqrt(r) = {1 / np.sqrt(r):.2f}")
print(f"Mediana de λ = {stats.gamma(r, scale=1 / alpha).median():.5f} por semana")
print(f"P(λ < 0.01/semana) = {stats.gamma(r, scale=1 / alpha).cdf(0.01):.3f}")
print(f"Prob. media de morir tras una compra E[p] = a/(a+b) = {media_p:.4f}")
print(f"Mediana de p = {stats.beta(a, b).median():.4f};  desv. est. de p = {stats.beta(a, b).std():.4f}")
print(f"Vida esperada (en compras repetidas) de un cliente típico ~ 1/E[p] = {1 / media_p:.1f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
lg = np.linspace(1e-4, 0.4, 600)
ax[0].plot(lg, stats.gamma(r, scale=1 / alpha).pdf(lg), color="C0")
ax[0].axvline(media_lam, color="k", ls="--", label=f"media = {media_lam:.3f}")
ax[0].set(title=f"λ ~ Gamma(r={r:.3f}, α={alpha:.3f})", xlabel="compras por semana",
          ylim=(0, 40))
ax[0].legend()
pg = np.linspace(1e-4, 1 - 1e-4, 600)
ax[1].plot(pg, stats.beta(a, b).pdf(pg), color="C3")
ax[1].axvline(media_p, color="k", ls="--", label=f"media = {media_p:.3f}")
ax[1].set(title=f"p ~ Beta(a={a:.3f}, b={b:.3f})", xlabel="prob. de morir tras una compra",
          ylim=(0, 6))
ax[1].legend()
guardar(fig, "2_2_densidades")

# %% [markdown]
# **2.1 Respuesta.** Obtenemos $(r,\alpha,a,b)=(0.2426, 4.4136, 0.7929,
# 2.4259)$ con log-verosimilitud $-9582.43$: idénticos al artículo (0.243,
# 4.414, 0.793, 2.426; $-9582.4$) al redondeo. Cinco de seis arranques llegan
# al mismo punto; el sexto se perdió en $a\to 0$, $b\to\infty$ con una
# log-verosimilitud 181 unidades peor — exactamente el "optimizador perdido"
# contra el que advierte el enunciado. Por eso siempre arrancamos desde varios
# puntos y nos quedamos con el mejor.
#
# **2.2 Respuesta.**
#
# * **Tasa de compra de un cliente activo:** $E[\lambda]=r/\alpha=0.055$
#   compras por semana, unas **2.9 compras repetidas al año**. Pero la media
#   engaña: con $r=0.24<1$ la gamma tiene forma de "L", con moda en 0 y cola
#   larga (CV = $1/\sqrt r$ = 2.0). La **mediana es 0.009** por semana (una
#   compra cada ~2 años) y la mitad de los clientes tiene $\lambda<0.01$. Unos
#   pocos clientes muy activos jalan la media hacia arriba.
# * **Probabilidad de morir después de una compra:** $E[p]=a/(a+b)=0.246$. En
#   promedio, uno de cada cuatro "se va" después de cada recompra, lo que da
#   una vida esperada de unas 4 compras repetidas. Con $a=0.79<1$ la densidad
#   de $p$ se dispara cerca de 0 (muchos clientes muy leales) pero tiene cola
#   larga hacia 1 (clientes que casi seguro se van tras la siguiente compra);
#   la desviación estándar (0.21) es casi igual a la media.
# * **¿Qué tan parecidos son los clientes?** Muy poco. Ambas distribuciones
#   están lejos de ser una punta alrededor de la media: hay muchísima
#   heterogeneidad tanto en *cuánto* compran como en *cuánto duran*. Por eso un
#   modelo que trata a todos igual (o la serie agregada) pierde información, y
#   por eso tiene sentido hablar de "quién sigue vivo" cliente por cliente.

# %% [markdown]
# ### 2.3 Bono: error estándar por *bootstrap* sobre clientes
#
# Remuestreamos los 2,357 clientes con reemplazo 300 veces y reestimamos.
# Como referencia agregamos también el error estándar asintótico (inversa del
# hessiano numérico en log-parámetros, llevado a la escala original con el
# método delta).

# %%
B = 300
rng = np.random.default_rng(2024)
boot = []
for _ in range(B):
    idx = rng.integers(0, len(x), len(x))
    fb = bgnbd.fit(x[idx], tx[idx], T[idx], starts=[ajuste["logparams"], np.zeros(4)])
    boot.append(fb["params"])
boot = pd.DataFrame(boot, columns=["r", "alpha", "a", "b"])


def hessiano(f, z, h=1e-4):
    k = len(z)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(z + ei + ej) - f(z + ei - ej) - f(z - ei + ej) + f(z - ei - ej)) / (4 * h * h)
    return H


negll = lambda z: -bgnbd.loglik(np.exp(z), x, tx, T)
cov_log = np.linalg.inv(hessiano(negll, ajuste["logparams"]))
se_hess = P * np.sqrt(np.diag(cov_log))
# desviación de la simulación 1.3 reescalada: CV de la simulación x estimado de CDNOW
se_sim = (est[["r", "alpha", "a", "b"]].std() / VERDAD).to_numpy() * P
tabla(pd.DataFrame({
    "estimado": P,
    "EE bootstrap": boot.std().to_numpy(),
    "EE hessiano": se_hess,
    "desv. simulación 1.3": est[["r", "alpha", "a", "b"]].std().to_numpy(),
    "CV bootstrap %": 100 * boot.std().to_numpy() / P,
    "CV simulación %": 100 * (est[["r", "alpha", "a", "b"]].std() / VERDAD).to_numpy(),
    "IC 95% bootstrap": [f"[{boot[c].quantile(.025):.3f}, {boot[c].quantile(.975):.3f}]"
                         for c in boot.columns],
}, index=["r", "alpha", "a", "b"]).round(4), "2_3_bootstrap")

# %% [markdown]
# **2.3 Respuesta.** El *bootstrap* da errores estándar de 0.014 para $r$,
# 0.45 para $\alpha$, 0.18 para $a$ y 0.71 para $b$. El patrón es el mismo de
# la simulación 1.3: $r$ y $\alpha$ con CV de 6–10%, $a$ y $b$ con CV de
# 22–29%. Para $a$ y $b$ los tres números (bootstrap, hessiano y la
# dispersión de la simulación) prácticamente coinciden, lo que dice que la
# simulación fue un buen ensayo del problema real.
#
# Para $r$ y sobre todo $\alpha$, el bootstrap es más grande que el hessiano
# (0.45 vs 0.38) y que la dispersión simulada. El error estándar del hessiano
# y la simulación suponen que **el modelo es el verdadero**; el bootstrap no.
# Que el bootstrap sea mayor sugiere que en los datos reales hay variabilidad
# en el proceso de compra que el BG/NBD no captura del todo (por ejemplo
# compras más regulares o más "en racimos" que un Poisson), es decir, cierta
# mala especificación. Con datos reales conviene preferir el bootstrap.

# %% [markdown]
# ---
# ## Parte 3: pronóstico de abajo hacia arriba y de arriba hacia abajo
#
# ### 3.1 *Bottom-up*
#
# El cliente $i$ hizo su primera compra en la semana $s_i$ (días/7). Las
# compras repetidas esperadas de toda la cohorte hasta la semana $k$ son
# $\sum_i \mathbb{E}[X(k - s_i)]$ (con $\mathbb{E}[X(t)] = 0$ si $t \le 0$).

# %%
trans = trans.merge(d[["primera"]], left_on="id", right_index=True)
trans["repetida"] = trans["fecha"] > trans["primera"]
trans["sem"] = trans["dia"] // 7 + 1          # semana 1..78
semanas = np.arange(1, 79)
real_sem = (trans[trans["repetida"]].groupby("sem").size()
            .reindex(semanas, fill_value=0).astype(float))
real_acum = real_sem.cumsum()
t0 = d["t0"].to_numpy()
bu_acum = pd.Series([bgnbd.expected_x(k - t0, P).sum() for k in semanas], index=semanas)
bu_sem = bu_acum.diff().fillna(bu_acum.iloc[0])

hold_real = real_acum[78] - real_acum[39]
hold_bu = bu_acum[78] - bu_acum[39]
print(f"Compras repetidas en el holdout (sem 40-78): real = {hold_real:.0f}, "
      f"BG/NBD = {hold_bu:.1f}  (error {100 * (hold_bu / hold_real - 1):+.1f}%)")
print(f"Acumulado a la semana 78: real = {real_acum[78]:.0f}, BG/NBD = {bu_acum[78]:.1f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(semanas, real_acum, "k", label="real")
ax[0].plot(semanas, bu_acum, "C0--", label="BG/NBD (bottom-up)")
ax[0].axvline(39, color="gray", ls=":")
ax[0].set(title="Compras repetidas acumuladas", xlabel="semana")
ax[0].legend()
ax[1].plot(semanas, real_sem, "k", label="real")
ax[1].plot(semanas, bu_sem, "C0--", label="BG/NBD")
ax[1].axvline(39, color="gray", ls=":")
ax[1].set(title="Compras repetidas por semana", xlabel="semana")
ax[1].legend()
guardar(fig, "3_1_bottom_up")

# %% [markdown]
# **3.1 Respuesta.** La curva acumulada del BG/NBD sigue muy de cerca a la
# real en calibración y se va quedando un poco abajo en el *holdout*. A la
# semana 78 pronostica 4,166 compras repetidas acumuladas contra 4,339 reales,
# un **subpronóstico de 4.0%, exactamente lo que reporta el artículo**
# ("under-forecasting by ... 4%"). Medido solo en el *holdout* (semanas
# 40–78) el modelo pronostica 1,664 contra 1,882 reales (−11.6%): el error
# relativo se ve más grande porque se calcula sobre un total más chico. En la
# serie semanal se ve por qué: la serie real tiene picos (semana 45, inicios
# de noviembre; semanas 49–50, la temporada navideña de diciembre; semanas
# 61–65, marzo de 1998) que una curva suave y sin estacionalidad no puede
# reproducir, y entre las semanas 45 y 66 la serie real se mantiene casi
# siempre por encima del modelo: la caída real es algo más lenta que la
# predicha.

# %% [markdown]
# ### 3.2 *Top-down*
#
# Ahora solo vemos la serie semanal agregada de las semanas 1–39 y
# pronosticamos 40–78 con métodos de la tarea de series de tiempo. Los que
# dependen de una ventana se prueban con varias.

# %%
y = real_sem.loc[1:39].to_numpy()
h = 39
semanas_h = np.arange(40, 79)


def tendencia(y, desde):
    t = np.arange(desde, 40)
    m, c = np.polyfit(t, y[desde - 1:], 1)
    return np.maximum(m * semanas_h + c, 0)


metodos = {
    "Media histórica (sem 1-39)": np.full(h, y.mean()),
    "Media histórica (sem 13-39)": np.full(h, y[12:].mean()),
    "Naive (última semana)": np.full(h, y[-1]),
    "Media últimas 4 sem": np.full(h, y[-4:].mean()),
    "Media últimas 13 sem": np.full(h, y[-13:].mean()),
    "Tendencia lineal (sem 1-39)": tendencia(y, 1),
    "Tendencia lineal (sem 13-39)": tendencia(y, 13),
    "Tendencia lineal (sem 26-39)": tendencia(y, 26),
    "Drift (sem 13-39)": np.maximum(y[-1] + (y[-1] - y[12]) / (39 - 13) * np.arange(1, h + 1), 0),
    "BG/NBD (bottom-up)": bu_sem.loc[40:78].to_numpy(),
}
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(semanas, real_sem, "k", lw=2, label="real")
for i, (nombre, f) in enumerate(metodos.items()):
    estilo = "-" if "BG" in nombre else "--"
    ax.plot(semanas_h, f, estilo, lw=2.5 if "BG" in nombre else 1.2, label=nombre)
ax.axvline(39.5, color="gray", ls=":")
ax.set(title="Pronóstico top-down vs bottom-up de compras repetidas semanales", xlabel="semana")
ax.legend(fontsize=7, ncol=2)
guardar(fig, "3_2_top_down")

# %% [markdown]
# ### 3.3 Comparación

# %%
real_h = real_sem.loc[40:78].to_numpy()
comp = pd.DataFrame({
    "total holdout": {k: f.sum() for k, f in metodos.items()},
    "error %": {k: 100 * (f.sum() / real_h.sum() - 1) for k, f in metodos.items()},
    "MAE semanal": {k: np.mean(np.abs(f - real_h)) for k, f in metodos.items()},
})
comp.loc["(real)"] = [real_h.sum(), 0, 0]
tabla(comp.sort_values("MAE semanal").round(2), "3_3_comparacion")

# %%
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.bar(semanas[:20], real_sem.iloc[:20], color="gray", label="compras repetidas (real)")
ax2 = ax.twinx()
nuevos = d.groupby(np.floor(d["t0"]).astype(int) + 1).size().reindex(semanas[:20], fill_value=0)
ax2.plot(semanas[:20], nuevos.cumsum(), "C3o-", label="clientes adquiridos (acum.)")
ax.set(xlabel="semana", ylabel="compras repetidas", title="Las primeras semanas: la cohorte se está formando")
ax2.set_ylabel("clientes en la cohorte")
fig.legend(loc="upper left", bbox_to_anchor=(0.08, 0.88))
guardar(fig, "3_3_primeras_semanas")

# %% [markdown]
# **3.3 Respuesta.**
#
# * **¿Quién gana?** Ex post, una sola variante top-down — la tendencia lineal
#   ajustada en las semanas 26–39 — tiene un MAE semanal ligeramente menor
#   (9.3 vs 9.8) y un error total menor (−7% vs −12%) que el BG/NBD. Pero el
#   BG/NBD es el segundo, y le gana a **todos** los demás métodos top-down.
# * **Sensibilidad a la ventana:** enorme. La misma tendencia lineal da +29%
#   con la ventana 1–39, −50% con la 13–39 y −7% con la 26–39; la media
#   histórica da +31%, la naive +28%, el drift −24%. Elegir la ventana
#   "ganadora" sabiendo el resultado es hacer trampa: el 30 de septiembre no
#   había forma de saber cuál de estas variantes, que van de −50% a +32%,
#   escoger. El BG/NBD no tiene ventanas que escoger y su error está dentro
#   de lo que se puede esperar con 2,357 clientes.
# * **Las primeras 12 semanas:** la cohorte **se está formando**. Los
#   clientes llegaron entre la semana 1 y la 12 (gráfica de arriba), así que
#   el número de clientes "en riesgo" de recomprar crece semana a semana y las
#   compras repetidas suben mecánicamente. A partir de la semana 13 la cohorte
#   está cerrada: ya no entra nadie y los que están se van muriendo, por eso la
#   serie baja. Un método top-down solo ve una serie que sube y luego baja; no
#   puede saber que la subida fue por adquisición, que la adquisición se
#   detuvo, ni cuántos clientes siguen vivos. El bottom-up usa la fecha de
#   primera compra de cada cliente y lo sabe por construcción.

# %% [markdown]
# ### 3.4 Por cliente: esperanzas condicionales

# %%
ey39 = bgnbd.conditional_expected(39, P, x, tx, T)
grupo = np.where(x >= 7, "7+", x.astype(int).astype(str))
cond = (pd.DataFrame({"x": grupo, "E[Y(39)]": ey39, "x* real": xs})
        .groupby("x").agg(clientes=("x* real", "size"),
                          **{"BG/NBD E[Y(39)]": ("E[Y(39)]", "mean"),
                             "real x*": ("x* real", "mean")}))
cond = cond.loc[["0", "1", "2", "3", "4", "5", "6", "7+"]]
cond["error %"] = 100 * (cond["BG/NBD E[Y(39)]"] / cond["real x*"] - 1)
print(f"Suma de E[Y(39)] sobre clientes = {ey39.sum():.1f} (real {xs.sum():.0f})")
print(f"Correlación entre x* real y E[Y(39)] por cliente = {np.corrcoef(xs, ey39)[0, 1]:.3f} "
      f"(el artículo reporta 0.626)")
print(f"Grupo x=0: {xs[x == 0].sum():.0f} compras en el holdout "
      f"({100 * xs[x == 0].sum() / xs.sum():.0f}% del total)")
tabla(cond.round(3), "3_4_condicional")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(cond.index, cond["real x*"], "ko-", label="real")
ax.plot(cond.index, cond["BG/NBD E[Y(39)]"], "C0s--", label="BG/NBD")
ax.set(xlabel="compras repetidas en calibración (x)", ylabel="compras promedio en holdout",
       title="Esperanzas condicionales")
ax.legend()
guardar(fig, "3_4_condicional")

# %% [markdown]
# **3.4 Respuesta.** El modelo reproduce la forma creciente de la relación
# (más compras pasadas, más compras futuras) y acierta muy bien en el gran
# grupo $x=0$ (0.225 pronosticado vs 0.237 real; el artículo reporta 0.23 vs
# 0.24) y en $x=3$ (1.52 vs 1.56, igual que en el artículo) y $7+$ (−3%).
# Donde **más se equivoca es en $x=1$ y $x=2$** (−25% en ambos): el modelo
# subestima a los clientes que recompraron una o dos veces; con $E[p]=0.25$
# "mata" a demasiados de ellos tras su última compra. También subestima a
# $x=4$ (−15%). Como casi todos los grupos quedan por debajo, el total (1,653)
# queda 12% abajo. La correlación entre $x^*$ real y $E[Y(39)]$ por cliente es
# 0.626, y el grupo $x=0$ hace 334 compras en el *holdout* (18% del total):
# ambos números son idénticos a los del artículo. Que el modelo acierte en
# el grupo silencioso es importante porque es el grupo más grande.
#
# **¿Esto lo puede hacer un top-down?** No. Un pronóstico top-down es de la
# serie agregada; no tiene idea de qué clientes componen el total, así que no
# puede decir cuánto va a comprar un cliente con $x=2$ comparado con uno con
# $x=0$, ni asignar el pronóstico a individuos. A lo sumo podría repartir el
# total proporcionalmente, lo cual supone que todos son iguales —
# exactamente lo que la Parte 2 mostró que no es cierto.

# %% [markdown]
