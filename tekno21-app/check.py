import os
import pandas as pd

m = pd.read_csv("samples/manifest.csv")
print("Columnas:", m.columns.tolist())
print(m.head(), "\n")
print(m["label_name"].value_counts(), "\n")

faltan = [f for f in m["file"] if not os.path.exists(f"samples/{f}")]
print("Imágenes que faltan:", faltan if faltan else "ninguna ✅")
print("Tiene pred_notebook:", "pred_notebook" in m.columns)
