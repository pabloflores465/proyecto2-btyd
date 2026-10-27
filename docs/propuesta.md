# Propuesta — Proyecto 2: Contando clientes con modelos *Buy 'Til You Die*

**Integrantes:** _(completar)_ · _(completar)_ · _(completar)_

## Quién se encarga de qué

| Parte | Responsable | Entregable |
|---|---|---|
| 0. Preparación de CDNOW (x, tₓ, T, x*) | _(integrante 1)_ | `btyd/data.py` |
| 1. Derivación 1.1–1.2, verosimilitud y simulación 1.3 | _(integrante 1 y 2)_ | `btyd/bgnbd.py`, `docs/checkpoint.pdf` |
| 2. Ajuste a CDNOW, interpretación, bootstrap | _(integrante 2)_ | notebook §2 |
| 3. Bottom-up vs top-down | _(integrante 3)_ | notebook §3 |
| 4. P(vivo), P(Z=1), MBG/NBD | _(integrante 1)_ | notebook §4, `btyd/mbgnbd.py` |
| 5. Online Retail | _(integrante 3)_ | notebook §5 |
| 6. Regresión logística (opcional) | _(integrante 2)_ | notebook §6 |
| Presentación y README | todos | `presentacion/presentacion.pdf` |

Todos revisan la derivación, porque cualquiera debe poder explicarla.

## Cómo vamos a hacer la simulación de 1.3

* **Parámetros:** $(r, \alpha, a, b) = (0.25, 4, 0.8, 2.5)$, cercanos a los que esperamos en CDNOW,
  2,357 clientes y $T_i \sim U(27, 39)$ semanas, imitando la adquisición escalonada de CDNOW.
* **Generación:** para cada cliente sorteamos $\lambda_i \sim \text{Gamma}(r, \alpha)$ y
  $p_i \sim \text{Beta}(a, b)$. A partir de la primera compra en $t = 0$, sorteamos tiempos entre
  compras exponenciales con tasa $\lambda_i$; después de cada compra repetida el cliente muere con
  probabilidad $p_i$. Registramos $x$, $t_x$ y $T$.
* **Semillas:** 50 semillas distintas (más que las 30 mínimas).
* **Estimación:** máxima verosimilitud sobre log-parámetros con L-BFGS-B (cotas ±10) desde 6
  arranques, pulido con Nelder–Mead.
* **¿Cómo sabremos si funcionó?** (1) la media de los estimados queda a menos de un error estándar
  del valor verdadero; (2) los intervalos ±1.96 desviaciones cubren el valor verdadero ≈95% de las
  veces; (3) ningún estimado queda pegado a las cotas; (4) además, comprobamos por Monte Carlo que
  la fórmula cerrada de la verosimilitud (1.2) coincide con la integral numérica de la de 1.1.
  Si el código no recupera los parámetros que nosotros pusimos, no lo usamos con datos reales.
