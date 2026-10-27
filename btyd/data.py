"""Lectura y preparación de CDNOW y Online Retail."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

CDNOW_INICIO = pd.Timestamp("1997-01-01")
CDNOW_FIN_CAL = pd.Timestamp("1997-09-30")
CDNOW_FIN_HOLD = pd.Timestamp("1998-06-30")


def leer_cdnow(path=DATA / "CDNOW_sample.txt"):
    """Lee CDNOW y junta las compras de un mismo cliente en un mismo día (0.1)."""
    df = pd.read_csv(path, sep=r"\s+", header=None,
                     names=["id_master", "id", "fecha", "cds", "monto"],
                     dtype={"fecha": str})
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y%m%d")
    df = (df.groupby(["id", "fecha"], as_index=False)
            .agg(cds=("cds", "sum"), monto=("monto", "sum")))
    df["dia"] = (df["fecha"] - CDNOW_INICIO).dt.days
    df["semana"] = df["dia"] / 7.0
    return df


def rfm(trans, fin_cal, fin_hold=None, col_id="id", col_fecha="fecha",
        inicio=CDNOW_INICIO):
    """Resumen (x, tx, T) por cliente con corte en fin_cal (inclusive).

    Solo entran los clientes con primera compra <= fin_cal. Si se da
    fin_hold, agrega x_star = compras en (fin_cal, fin_hold]. El tiempo se
    mide en semanas = días / 7.
    """
    fin_cal = pd.Timestamp(fin_cal)
    cal = trans[trans[col_fecha] <= fin_cal]
    g = cal.groupby(col_id)[col_fecha]
    primera = g.min()
    ultima = g.max()
    out = pd.DataFrame({
        "primera": primera,
        "x": g.count() - 1,
        "tx": (ultima - primera).dt.days / 7.0,
        "T": (fin_cal - primera).dt.days / 7.0,
        "t0": (primera - inicio).dt.days / 7.0,
    })
    if fin_hold is not None:
        hold = trans[(trans[col_fecha] > fin_cal) & (trans[col_fecha] <= pd.Timestamp(fin_hold))]
        out["x_star"] = hold.groupby(col_id).size().reindex(out.index, fill_value=0)
    return out


def cdnow_rfm():
    return rfm(leer_cdnow(), CDNOW_FIN_CAL, CDNOW_FIN_HOLD)


# ---------------------------------------------------------------------------
# Online Retail
# ---------------------------------------------------------------------------
RETAIL_FIN_CAL = pd.Timestamp("2011-06-30")
RETAIL_FIN_HOLD = pd.Timestamp("2011-12-08")

# Códigos que no son productos: envío, ajustes, comisiones, muestras, etc.
NO_PRODUCTO = {"POST", "DOT", "M", "m", "D", "C2", "S", "B", "CRUK", "PADS",
               "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2", "TEST001",
               "TEST002", "gift_0001_10", "gift_0001_20", "gift_0001_30",
               "gift_0001_40", "gift_0001_50"}


def leer_retail(path=DATA / "Online Retail.xlsx"):
    """Lee Online Retail, usando un caché en pickle si existe (y bajándolo de UCI si falta)."""
    cache = DATA / "online_retail.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    if not Path(path).exists():
        _descargar_retail(path)
    df = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
    df.to_pickle(cache)
    return df


def _descargar_retail(path):
    """Baja Online Retail del repositorio UCI si no está en data/."""
    import io
    import urllib.request
    import zipfile
    url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    with urllib.request.urlopen(url) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extract("Online Retail.xlsx", Path(path).parent)


def limpiar(df):
    """Limpieza de Online Retail (la misma lógica de la tarea de series).

    - quita duplicados exactos,
    - quita códigos que no son productos (envío, comisiones, ajustes...),
    - quita precios <= 0,
    - conserva las devoluciones (facturas 'C') para poder calcular venta neta,
    - agrega la venta de la línea y la fecha (sin hora).
    """
    df = df.drop_duplicates().copy()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df = df[~df["StockCode"].isin(NO_PRODUCTO)]
    df = df[df["UnitPrice"] > 0]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["fecha"] = df["InvoiceDate"].dt.normalize()
    df["venta"] = df["Quantity"] * df["UnitPrice"]
    df["cancelada"] = df["InvoiceNo"].astype(str).str.startswith("C")
    return df


def retail_transacciones(df):
    """Una transacción = un día con venta neta positiva de un cliente."""
    df = df[df["CustomerID"].notna()].copy()
    df["CustomerID"] = df["CustomerID"].astype(int)
    dia = df.groupby(["CustomerID", "fecha"], as_index=False)["venta"].sum()
    return dia[dia["venta"] > 0].rename(columns={"CustomerID": "id"})


def retail_rfm():
    trans = retail_transacciones(limpiar(leer_retail()))
    inicio = trans["fecha"].min()
    return rfm(trans, RETAIL_FIN_CAL, RETAIL_FIN_HOLD, inicio=inicio), trans
