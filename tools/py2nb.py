"""Convierte un script en formato 'percent' (# %% / # %% [markdown]) a .ipynb."""
import re
import sys

import nbformat


def convertir(src, dst):
    texto = open(src, encoding="utf-8").read()
    bloques = re.split(r"^# %%(.*)$", texto, flags=re.M)
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    # bloques = [preámbulo, marca1, cuerpo1, marca2, cuerpo2, ...]
    for marca, cuerpo in zip(bloques[1::2], bloques[2::2]):
        cuerpo = cuerpo.strip("\n")
        if "[markdown]" in marca:
            lineas = [re.sub(r"^# ?", "", l) for l in cuerpo.splitlines()]
            nb.cells.append(nbformat.v4.new_markdown_cell("\n".join(lineas)))
        elif cuerpo.strip():
            nb.cells.append(nbformat.v4.new_code_cell(cuerpo))
    nbformat.write(nb, dst)


if __name__ == "__main__":
    convertir(sys.argv[1], sys.argv[2])
