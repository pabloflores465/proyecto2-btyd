# Proyecto 2: Contando clientes con modelos *Buy 'Til You Die*

Análisis y Ciencia de Datos, Facultad de Ingeniería, Universidad del Istmo de Guatemala.

Construimos el **BG/NBD** (Fader, Hardie y Lee, 2005) desde cero: derivamos la verosimilitud, la
programamos a mano con `scipy.special` (`gammaln`, `betaln`, `hyp2f1`), la estimamos por máxima
verosimilitud con `scipy.optimize.minimize` y la usamos para pronosticar compras y probabilidades de
compra en CDNOW y Online Retail. No usamos `lifetimes` ni `PyMC-Marketing`; validamos contra el
artículo original y por simulación.

## Estructura

```
proyecto2.ipynb         Notebook principal, ya ejecutado (Partes 0–6, con respuestas)
proyecto2.py            El mismo notebook en formato script (# %%), corre de principio a fin
btyd/
  bgnbd.py              BG/NBD: log-verosimilitud, ajuste, E[X(t)], E[Y(τ)], P(vivo), P(Z=1), simulador
  mbgnbd.py             MBG/NBD (Parte 4.6)
  data.py               Lectura de los archivos originales, limpieza y resumen (x, tx, T, x*)
data/                   CDNOW_sample.txt (Hardie) y Online Retail.xlsx (UCI), sin modificar
figuras/                Gráficas generadas por el notebook
resultados/             Tablas generadas por el notebook (.csv)
presentacion/           presentacion_historia.pptx (27 diapositivas en 5 actos, con notas del orador),
                        presentacion.pdf (versión Beamer) y su fuente LaTeX
docs/
  checkpoint.pdf        Derivaciones 1.1, 1.2 y 4.3, simulación 1.3 y estimados 2.1
  propuesta.md          Propuesta (integrantes, roles, plan de simulación)
tools/py2nb.py          Convierte proyecto2.py en proyecto2.ipynb
```

## Cómo correrlo

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace proyecto2.ipynb   # o: python proyecto2.py
```

Tarda alrededor de 1 minuto (más la primera vez, porque lee el Excel de Online Retail y lo guarda
en caché). Si `data/Online Retail.xlsx` no está, se descarga de UCI. Para regenerar el notebook a
partir del script: `python tools/py2nb.py proyecto2.py proyecto2.ipynb`.

## Resultados principales

| | Nuestro | Artículo |
|---|---|---|
| $(r, \alpha, a, b)$ | (0.2426, 4.4136, 0.7929, 2.4259) | (0.243, 4.414, 0.793, 2.426) |
| Log-verosimilitud | −9582.43 | −9582.4 |
| Compras repetidas acumuladas, semana 78 | −4.0% | −4% |
| Corr(x\*, E[Y(39)]) | 0.626 | 0.626 |

* **1.3 Simulación (50 semillas):** se recuperan los parámetros; r y α con CV de 4–7%, a y b de
  24–29%, porque los clientes con x = 0 no informan nada sobre la muerte.
* **3. Bottom-up vs top-down:** el BG/NBD pronostica el *holdout* en −12% sin escoger ventanas; los
  métodos top-down (medias, naive, drift, tendencias lineal y logarítmica) van de −50% a +53% según la ventana.
* **4. ¿Quién sigue vivo?** P(vivo) como clasificador tiene AUC 0.32, peor que el azar, porque
  P(vivo) = 1 para todos los clientes con x = 0. La probabilidad correcta, P(Z = 1), tiene AUC 0.757,
  igual que la recencia sola (0.763), pero está calibrada (Brier 0.161 vs 0.206). El MBG/NBD da
  P(vivo) = 0.41 para x = 0 y casi el mismo P(Z = 1).
* **5. Online Retail:** a → 0 y b → ∞: nadie se muere y el modelo colapsa al NBD (mayoristas,
  calibración corta, truncamiento por la izquierda). Subpronostica el *holdout* en 13%, y noviembre
  en 38%, por la estacionalidad.
* **6. Logística:** solo le gana al BG/NBD cuando se entrena con etiquetas del futuro; en la
  versión honesta ordenan parecido y el BG/NBD tiene mejor Brier.

## Supuestos y decisiones

* Compras del mismo cliente el mismo día cuentan como una sola transacción; el tiempo se mide en
  semanas = días/7 con el día 0 = 1-ene-1997.
* **Online Retail:** la función `limpiar` (`btyd/data.py`) reproduce la limpieza de la tarea de
  series de tiempo: quita duplicados, códigos que no son productos (POST, DOT, M, D, BANK CHARGES,
  AMAZONFEE, etc.) y precios ≤ 0, y conserva devoluciones para calcular venta neta. Una transacción
  es un día con venta neta positiva de un cliente con `CustomerID`. Si su función de la tarea
  difiere, basta con reemplazar `limpiar`.

## Referencias

Fader, Hardie y Lee (2005), *Marketing Science* 24(2) · Schmittlein, Morrison y Colombo (1987) ·
Batislam, Denizel y Filiztekin (2007) · Fader y Hardie (2009) · Hyndman y Athanasopoulos, *FPP2* ·
Datos: [Bruce Hardie](https://www.brucehardie.com/datasets/),
[UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail).
