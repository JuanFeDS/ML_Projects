"""Corrige el nombre del archivo de salida en NB03 cell[42]."""
import json

path = (
    r"C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects"
    r"\Projects\Classification\prueba_plantilla_de_datos\notebooks\exploratory"
    r"\03.feature_engineering.ipynb"
)

with open(path, encoding="utf-8") as f:
    nb = json.load(f)

# Fix cell [42]: change train_scaled.csv → train_features_scaled.csv
cell = nb["cells"][42]
src = "".join(cell["source"])
src = src.replace("train_scaled.csv", "train_features_scaled.csv")
cell["source"] = [src]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("NB03 cell[42] corregido: train_scaled.csv -> train_features_scaled.csv")
