"""Crea el Notebook 04 — Model Training para Spaceship Titanic."""
import json
import uuid

output = (
    r"C:\Users\jmart\Documents\Proyectos\Data_Science\06_Proyectos\ML_Projects"
    r"\Projects\Classification\prueba_plantilla_de_datos\notebooks\exploratory"
    r"\04.Model_Training.ipynb"
)


def cid():
    return str(uuid.uuid4())[:8]


def md(source_lines):
    return {
        "cell_type": "markdown",
        "id": cid(),
        "metadata": {},
        "source": source_lines,
    }


def code(source_lines):
    return {
        "cell_type": "code",
        "id": cid(),
        "metadata": {},
        "source": source_lines,
        "outputs": [],
        "execution_count": None,
    }


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        # ── 0: Header
        md([
            "# **Model Training — Spaceship Titanic**\n",
            "\n",
            "**Objetivo:** Entrenar, comparar y evaluar modelos de clasificación sobre el dataset preparado en NB03.\n",
            "\n",
            "**Cadena de análisis previa:**\n",
            "\n",
            "| Notebook | Contribución clave |\n",
            "|---|---|\n",
            "| [NB01 — Exploración](01.Initial_exploration.ipynb) | 8,693 registros · 14 columnas · balanceado 50/50 · 0 duplicados |\n",
            "| [NB02 — Análisis vs Target](02.Analisis_Target.ipynb) | CryoSleep (chi²=1859), Deck (chi²=392), TotalSpending_Log (r=-0.469) como features clave |\n",
            "| [NB03 — Feature Engineering](03.feature_engineering.ipynb) | 8,514 filas × 35 features · escalado · guardado en `data/processed/` |\n",
            "\n",
            "**Dataset de entrada:** `data/processed/train_features_scaled.csv`  \n",
            "**Métrica principal:** Accuracy (dataset balanceado 50/50 → válida y directamente interpretable)\n",
        ]),

        # ── 1: Librerías
        md(["## **1. Librerías**\n"]),
        code([
            "import sys\n",
            "sys.path.insert(0, '../../')\n",
            "\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import plotly.express as px\n",
            "import plotly.graph_objects as go\n",
            "from plotly.subplots import make_subplots\n",
            "\n",
            "from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV\n",
            "from sklearn.dummy import DummyClassifier\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, classification_report,\n",
            "    confusion_matrix, roc_auc_score,\n",
            ")\n",
            "import joblib\n",
            "import json\n",
            "import os\n",
            "import warnings\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "pd.set_option('display.max_columns', None)\n",
        ]),

        # ── 2: Cargar datos
        md(["## **2. Cargar Dataset Procesado**\n"]),
        code([
            "# Dataset producido por NB03: escalado, encoded, con todas las features de NB02\n",
            "data_path = '../../data/processed/train_features_scaled.csv'\n",
            "df = pd.read_csv(data_path)\n",
            "\n",
            "target = 'Transported'\n",
            "X = df.drop(columns=[target])\n",
            "y = df[target]\n",
            "\n",
            "print(f'Dataset cargado: {X.shape[0]:,} filas x {X.shape[1]} features')\n",
            "print(f'Target balance: {y.mean():.1%} True | {1-y.mean():.1%} False')\n",
            "print(f'\\nPrimeras features: {X.columns.tolist()[:8]}')\n",
        ]),

        # ── 3: Estrategia de validación
        md([
            "## **3. Estrategia de Validación**\n",
            "\n",
            "- **Train/Validation split:** 80/20 con `stratify=y`\n",
            "- **Cross-validation:** StratifiedKFold (5 folds) para comparación de modelos\n",
            "- **Métrica principal:** Accuracy (dataset balanceado → directamente interpretable)\n",
            "- **Métricas secundarias:** ROC-AUC, F1-macro\n",
        ]),
        code([
            "X_train, X_val, y_train, y_val = train_test_split(\n",
            "    X, y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "\n",
            "print(f'Train : {X_train.shape[0]:,} filas')\n",
            "print(f'Val   : {X_val.shape[0]:,} filas')\n",
            "print(f'Train target balance: {y_train.mean():.1%}')\n",
            "print(f'Val   target balance: {y_val.mean():.1%}')\n",
        ]),

        # ── 4: Baseline
        md([
            "## **4. Modelo Baseline**\n",
            "\n",
            "Referencia mínima que cualquier modelo real debe superar.\n",
        ]),
        code([
            "baseline = DummyClassifier(strategy='most_frequent', random_state=42)\n",
            "baseline.fit(X_train, y_train)\n",
            "\n",
            "baseline_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring='accuracy')\n",
            "print(f'Baseline CV accuracy: {baseline_cv.mean():.4f} +/- {baseline_cv.std():.4f}')\n",
            "print(f'Baseline val accuracy: {accuracy_score(y_val, baseline.predict(X_val)):.4f}')\n",
        ]),

        # ── 5: Modelos candidatos
        md([
            "## **5. Comparación de Modelos Candidatos**\n",
            "\n",
            "Tres familias con hiperparámetros por defecto para identificar el mejor punto de partida:\n",
            "- **Logistic Regression:** baseline interpretable, indica si el problema es lineal\n",
            "- **Random Forest:** robusto ante no-linealidades, provee feature importance\n",
            "- **Gradient Boosting:** generalmente el más preciso en tabular data\n",
        ]),
        code([
            "models = {\n",
            "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
            "    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),\n",
            "    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),\n",
            "}\n",
            "\n",
            "results = {}\n",
            "for name, model in models.items():\n",
            "    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')\n",
            "    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')\n",
            "    results[name] = {\n",
            "        'cv_acc': cv_acc.mean(), 'cv_acc_std': cv_acc.std(),\n",
            "        'cv_auc': cv_auc.mean(),\n",
            "    }\n",
            "    print(f'{name:25s}  acc={cv_acc.mean():.4f} +/-{cv_acc.std():.4f}  auc={cv_auc.mean():.4f}')\n",
        ]),
        code([
            "results_df = pd.DataFrame(results).T.sort_values('cv_acc', ascending=False)\n",
            "\n",
            "fig = px.bar(\n",
            "    results_df.reset_index().rename(columns={'index': 'Modelo'}),\n",
            "    x='Modelo', y='cv_acc', error_y='cv_acc_std',\n",
            "    title='CV Accuracy por Modelo (5-fold)',\n",
            "    labels={'cv_acc': 'CV Accuracy'},\n",
            "    color='cv_acc', color_continuous_scale='Blues',\n",
            ")\n",
            "fig.add_hline(y=baseline_cv.mean(), line_dash='dash',\n",
            "              annotation_text=f'Baseline: {baseline_cv.mean():.3f}')\n",
            "fig.update_layout(height=400, showlegend=False)\n",
            "fig.show()\n",
        ]),

        # ── 6: Tuning
        md([
            "## **6. Tuning del Mejor Modelo**\n",
            "\n",
            "GridSearchCV sobre el modelo con mejor CV accuracy de la sección anterior.\n",
        ]),
        code([
            "best_model_name = results_df.index[0]\n",
            "print(f'Modelo seleccionado para tuning: {best_model_name}')\n",
            "\n",
            "if 'Random Forest' in best_model_name:\n",
            "    param_grid = {\n",
            "        'n_estimators': [100, 200],\n",
            "        'max_depth': [None, 10, 20],\n",
            "        'min_samples_split': [2, 5],\n",
            "    }\n",
            "    base_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)\n",
            "elif 'Gradient' in best_model_name:\n",
            "    param_grid = {\n",
            "        'n_estimators': [100, 200],\n",
            "        'max_depth': [3, 5],\n",
            "        'learning_rate': [0.05, 0.1],\n",
            "    }\n",
            "    base_estimator = GradientBoostingClassifier(random_state=42)\n",
            "else:\n",
            "    param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}\n",
            "    base_estimator = LogisticRegression(max_iter=1000, random_state=42)\n",
            "\n",
            "grid_search = GridSearchCV(\n",
            "    base_estimator, param_grid, cv=cv,\n",
            "    scoring='accuracy', n_jobs=-1, verbose=1,\n",
            ")\n",
            "grid_search.fit(X_train, y_train)\n",
            "\n",
            "print(f'\\nMejores hiperparámetros: {grid_search.best_params_}')\n",
            "print(f'Mejor CV accuracy     : {grid_search.best_score_:.4f}')\n",
            "best_model = grid_search.best_estimator_\n",
        ]),

        # ── 7: Evaluación final
        md(["## **7. Evaluación Final**\n"]),
        code([
            "y_pred  = best_model.predict(X_val)\n",
            "y_proba = best_model.predict_proba(X_val)[:, 1]\n",
            "\n",
            "print('=' * 50)\n",
            "print('EVALUACIÓN EN VALIDATION SET')\n",
            "print('=' * 50)\n",
            "print(f'Accuracy : {accuracy_score(y_val, y_pred):.4f}')\n",
            "print(f'ROC-AUC  : {roc_auc_score(y_val, y_proba):.4f}')\n",
            "print()\n",
            "print(classification_report(y_val, y_pred))\n",
        ]),
        code([
            "cm = confusion_matrix(y_val, y_pred)\n",
            "fig = px.imshow(\n",
            "    cm, text_auto=True, color_continuous_scale='Blues',\n",
            "    labels={'x': 'Prediccion', 'y': 'Real'},\n",
            "    x=['False', 'True'], y=['False', 'True'],\n",
            "    title='Confusion Matrix — Validation Set',\n",
            ")\n",
            "fig.update_layout(height=400, width=500)\n",
            "fig.show()\n",
        ]),

        # ── Feature importance
        md([
            "### **7.1 Feature Importance**\n",
            "\n",
            "> Verificamos que las features de mayor poder discriminativo en NB02\n",
            "> (CryoSleep, Deck, TotalSpending_Log) aparezcan entre las más importantes del modelo.\n",
        ]),
        code([
            "if hasattr(best_model, 'feature_importances_'):\n",
            "    fi = pd.DataFrame({\n",
            "        'feature': X.columns,\n",
            "        'importance': best_model.feature_importances_,\n",
            "    }).sort_values('importance', ascending=False).head(20)\n",
            "elif hasattr(best_model, 'coef_'):\n",
            "    fi = pd.DataFrame({\n",
            "        'feature': X.columns,\n",
            "        'importance': abs(best_model.coef_[0]),\n",
            "    }).sort_values('importance', ascending=False).head(20)\n",
            "\n",
            "fig = px.bar(\n",
            "    fi, x='importance', y='feature', orientation='h',\n",
            "    title='Top 20 Features por Importancia',\n",
            "    labels={'importance': 'Importancia', 'feature': 'Feature'},\n",
            "    color='importance', color_continuous_scale='Viridis',\n",
            ")\n",
            "fig.update_layout(height=600, showlegend=False)\n",
            "fig.show()\n",
        ]),

        # ── 8: Guardar modelo
        md(["## **8. Guardar Modelo**\n"]),
        code([
            "model_dir = '../../models'\n",
            "os.makedirs(model_dir, exist_ok=True)\n",
            "\n",
            "model_path = f'{model_dir}/best_model.pkl'\n",
            "joblib.dump(best_model, model_path)\n",
            "\n",
            "metadata = {\n",
            "    'model': best_model_name,\n",
            "    'params': grid_search.best_params_,\n",
            "    'cv_accuracy': round(grid_search.best_score_, 4),\n",
            "    'val_accuracy': round(accuracy_score(y_val, y_pred), 4),\n",
            "    'val_roc_auc': round(roc_auc_score(y_val, y_proba), 4),\n",
            "    'n_features': int(X.shape[1]),\n",
            "    'n_train_samples': int(X_train.shape[0]),\n",
            "}\n",
            "with open(f'{model_dir}/model_metadata.json', 'w') as f:\n",
            "    json.dump(metadata, f, indent=2)\n",
            "\n",
            "print(f'Modelo guardado: {model_path}')\n",
            "print(f'Metadatos: {metadata}')\n",
        ]),

        # ── 9: Resumen
        md(["## **9. Resumen**\n"]),
        code([
            "print('=' * 60)\n",
            "print('RESUMEN DE ENTRENAMIENTO')\n",
            "print('=' * 60)\n",
            "print(f'  Modelo            : {best_model_name}')\n",
            "print(f'  Hiperparámetros   : {grid_search.best_params_}')\n",
            "print(f'  CV Accuracy 5fold : {grid_search.best_score_:.4f}')\n",
            "print(f'  Val Accuracy      : {accuracy_score(y_val, y_pred):.4f}')\n",
            "print(f'  Val ROC-AUC       : {roc_auc_score(y_val, y_proba):.4f}')\n",
            "print(f'  Baseline (dummy)  : {baseline_cv.mean():.4f}')\n",
            "print(f'  Ganancia vs base  : +{accuracy_score(y_val, y_pred) - baseline_cv.mean():.4f}')\n",
            "print('=' * 60)\n",
            "print()\n",
            "print('Modelo listo → ver src/models/ para integración en pipeline')\n",
        ]),
    ],
}

with open(output, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"NB4 creado: {output}")
print(f"Celdas: {len(nb['cells'])}")
