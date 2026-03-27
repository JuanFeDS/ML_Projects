# 📊 EDA Template - [Nombre del Dataset]

**Fecha**: YYYY-MM-DD  
**Autor**: [Tu nombre]  
**Dataset**: [Nombre y versión del dataset]  
**Objetivo**: [Breve descripción del objetivo del análisis]

---

## 1. Resumen Ejecutivo

### Hallazgos Principales
- [ ] Hallazgo 1
- [ ] Hallazgo 2
- [ ] Hallazgo 3

### Recomendaciones
- [ ] Recomendación 1
- [ ] Recomendación 2

---

## 2. Información del Dataset

### Fuente de Datos
- **Origen**: [Base de datos, archivo, API, etc.]
- **Fecha de extracción**: YYYY-MM-DD
- **Período de datos**: YYYY-MM-DD a YYYY-MM-DD
- **Frecuencia de actualización**: [Diaria, semanal, mensual, etc.]

### Dimensiones
- **Número de filas**: X,XXX
- **Número de columnas**: XX
- **Tamaño en memoria**: XX MB
- **Duplicados**: X (X%)

### Variables

| Variable | Tipo | Descripción | Valores Únicos | Missing (%) |
|----------|------|-------------|----------------|-------------|
| var1     | int  | ...         | XXX            | X%          |
| var2     | str  | ...         | XXX            | X%          |
| ...      | ...  | ...         | ...            | ...         |

---

## 3. Calidad de Datos

### 3.1 Valores Faltantes

**Resumen**:
- Total de valores faltantes: X,XXX (X%)
- Variables con > 20% missing: [lista]

**Análisis por variable**:

```python
# Código para generar visualización
import missingno as msno
msno.matrix(df)
msno.heatmap(df)
```

**Patrón de missing**:
- [ ] Aleatorio (MCAR)
- [ ] Aleatorio condicional (MAR)
- [ ] No aleatorio (MNAR)

**Estrategia de imputación**:
- `variable_1`: [Método y justificación]
- `variable_2`: [Método y justificación]

### 3.2 Duplicados

- **Duplicados exactos**: X filas (X%)
- **Duplicados en ID**: X filas (X%)

**Acción**: [Eliminar, investigar, mantener]

### 3.3 Valores Atípicos

**Variables con outliers detectados**:

| Variable | Método | Outliers | % | Acción |
|----------|--------|----------|---|--------|
| var1     | IQR    | XXX      | X%| [Mantener/Eliminar/Investigar] |
| var2     | Z-score| XXX      | X%| [Mantener/Eliminar/Investigar] |

**Visualización**:
```python
# Boxplots
df[numerical_cols].plot(kind='box', subplots=True, layout=(3,3), figsize=(15,10))
```

### 3.4 Inconsistencias

**Detectadas**:
- [ ] Inconsistencia 1: [Descripción]
- [ ] Inconsistencia 2: [Descripción]

**Acciones correctivas**:
- [Acción 1]
- [Acción 2]

---

## 4. Análisis Univariado

### 4.1 Variables Numéricas

Para cada variable numérica clave:

#### Variable: [nombre_variable]

**Estadísticas descriptivas**:
```
count:    X,XXX
mean:     XXX.XX
std:      XXX.XX
min:      XXX.XX
25%:      XXX.XX
50%:      XXX.XX
75%:      XXX.XX
max:      XXX.XX
```

**Distribución**:
- Tipo: [Normal, Sesgada a derecha/izquierda, Bimodal, etc.]
- Skewness: X.XX
- Kurtosis: X.XX

**Visualización**:
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histograma
df['variable'].hist(bins=30, ax=axes[0])
axes[0].set_title('Distribución de variable')

# Q-Q plot
stats.probplot(df['variable'], dist="norm", plot=axes[1])
```

**Insights**:
- [Insight 1]
- [Insight 2]

**Transformaciones sugeridas**:
- [ ] Log transform
- [ ] Box-Cox
- [ ] Estandarización
- [ ] Ninguna

---

### 4.2 Variables Categóricas

Para cada variable categórica clave:

#### Variable: [nombre_variable]

**Frecuencias**:

| Categoría | Frecuencia | Porcentaje |
|-----------|------------|------------|
| Cat1      | X,XXX      | XX%        |
| Cat2      | X,XXX      | XX%        |
| Cat3      | X,XXX      | XX%        |

**Visualización**:
```python
# Gráfico de barras
df['variable'].value_counts().plot(kind='bar')
plt.title('Distribución de variable')
```

**Insights**:
- [Insight 1]
- [Insight 2]

**Desbalance**:
- Categoría mayoritaria: XX%
- Categoría minoritaria: XX%
- **Acción**: [Balanceo necesario / No necesario]

---

## 5. Análisis Bivariado

### 5.1 Relación con Variable Objetivo

#### Numéricas vs. Target

**Correlaciones con target**:

| Variable | Correlación | P-value | Significancia |
|----------|-------------|---------|---------------|
| var1     | 0.XX        | 0.XXX   | ***           |
| var2     | 0.XX        | 0.XXX   | **            |
| var3     | 0.XX        | 0.XXX   | NS            |

**Visualización**:
```python
# Boxplots por clase
for col in numerical_cols:
    df.boxplot(column=col, by='target')
    plt.title(f'{col} por Target')
```

**Insights**:
- [Insight 1]
- [Insight 2]

#### Categóricas vs. Target

**Chi-cuadrado tests**:

| Variable | Chi2 | P-value | Cramer's V | Significancia |
|----------|------|---------|------------|---------------|
| var1     | XXX  | 0.XXX   | 0.XX       | ***           |
| var2     | XXX  | 0.XXX   | 0.XX       | **            |

**Visualización**:
```python
# Stacked bar charts
pd.crosstab(df['variable'], df['target'], normalize='index').plot(kind='bar', stacked=True)
```

**Insights**:
- [Insight 1]
- [Insight 2]

### 5.2 Relaciones entre Features

**Matriz de correlación**:
```python
# Heatmap de correlaciones
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
```

**Multicolinealidad detectada**:

| Variable 1 | Variable 2 | Correlación | Acción |
|------------|------------|-------------|--------|
| var1       | var2       | 0.XX        | [Eliminar una / Combinar / Mantener] |

**VIF (Variance Inflation Factor)**:

| Variable | VIF  | Interpretación |
|----------|------|----------------|
| var1     | X.XX | [Bajo/Moderado/Alto] |
| var2     | X.XX | [Bajo/Moderado/Alto] |

---

## 6. Análisis Multivariado

### 6.1 PCA (Principal Component Analysis)

**Varianza explicada**:
- PC1: XX%
- PC2: XX%
- PC3: XX%
- Total (3 componentes): XX%

**Visualización**:
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(df[numerical_cols])

plt.scatter(components[:, 0], components[:, 1], c=df['target'])
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
```

**Insights**:
- [Insight 1]
- [Insight 2]

### 6.2 Clustering Exploratorio

**Método**: [K-means, DBSCAN, Hierarchical]
**Número óptimo de clusters**: X (método: [Elbow, Silhouette])

**Visualización**:
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=X)
clusters = kmeans.fit_predict(df[numerical_cols])

plt.scatter(components[:, 0], components[:, 1], c=clusters)
```

**Insights**:
- [Insight 1]
- [Insight 2]

---

## 7. Análisis Temporal (si aplica)

### 7.1 Tendencias

**Variables analizadas**: [lista]

**Tendencia general**:
- [ ] Creciente
- [ ] Decreciente
- [ ] Estacionaria
- [ ] Cíclica

**Visualización**:
```python
df.set_index('fecha')['variable'].plot(figsize=(12, 4))
plt.title('Tendencia temporal de variable')
```

### 7.2 Estacionalidad

**Detectada**: [Sí/No]
**Período**: [Diario, Semanal, Mensual, Anual]

**Visualización**:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['variable'], model='additive', period=X)
decomposition.plot()
```

### 7.3 Autocorrelación

**ACF/PACF**:
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['variable'], ax=axes[0])
plot_pacf(df['variable'], ax=axes[1])
```

---

## 8. Feature Engineering Sugerido

### 8.1 Nuevas Features

| Feature Propuesta | Fórmula/Lógica | Justificación |
|-------------------|----------------|---------------|
| feature_1         | [fórmula]      | [justificación] |
| feature_2         | [fórmula]      | [justificación] |

### 8.2 Transformaciones

| Variable Original | Transformación | Justificación |
|-------------------|----------------|---------------|
| var1              | log(x)         | [justificación] |
| var2              | sqrt(x)        | [justificación] |

### 8.3 Encoding

| Variable | Método | Justificación |
|----------|--------|---------------|
| var1     | One-Hot| [justificación] |
| var2     | Label  | [justificación] |
| var3     | Target | [justificación] |

---

## 9. Segmentación de Datos

### 9.1 Estrategia de Split

**Método**: [Random, Stratified, Time-based]
**Proporción**:
- Train: XX%
- Validation: XX%
- Test: XX%

**Justificación**: [Explicación de por qué esta estrategia]

### 9.2 Validación de Split

**Distribución de target**:

| Set        | Clase 0 | Clase 1 | Total |
|------------|---------|---------|-------|
| Train      | XX%     | XX%     | X,XXX |
| Validation | XX%     | XX%     | X,XXX |
| Test       | XX%     | XX%     | X,XXX |

**Verificación**: [OK / Ajustar]

---

## 10. Consideraciones para Modelado

### 10.1 Desbalance de Clases

**Ratio**: X:1 (mayoría:minoría)
**Severidad**: [Leve/Moderado/Severo]

**Estrategias sugeridas**:
- [ ] SMOTE
- [ ] Undersampling
- [ ] Class weights
- [ ] Ensemble methods

### 10.2 Features Recomendadas

**Top features por importancia**:
1. feature_1 (justificación)
2. feature_2 (justificación)
3. feature_3 (justificación)

**Features a eliminar**:
- feature_x: [razón]
- feature_y: [razón]

### 10.3 Algoritmos Sugeridos

Basado en el análisis:
- [ ] Logistic Regression (baseline)
- [ ] Random Forest
- [ ] XGBoost
- [ ] LightGBM
- [ ] Neural Network

**Justificación**: [Explicación]

---

## 11. Riesgos y Limitaciones

### 11.1 Calidad de Datos

- [ ] Riesgo 1: [Descripción y mitigación]
- [ ] Riesgo 2: [Descripción y mitigación]

### 11.2 Sesgos Detectados

- [ ] Sesgo 1: [Descripción]
- [ ] Sesgo 2: [Descripción]

**Implicaciones éticas**: [Discusión]

### 11.3 Limitaciones del Análisis

- Limitación 1
- Limitación 2

---

## 12. Próximos Pasos

- [ ] Paso 1: [Descripción]
- [ ] Paso 2: [Descripción]
- [ ] Paso 3: [Descripción]

---

## 13. Anexos

### Código Completo

Link al notebook: `notebooks/eda_[nombre]_[fecha].ipynb`

### Referencias

- [Referencia 1]
- [Referencia 2]

---

**Revisado por**: [Nombre]  
**Fecha de revisión**: YYYY-MM-DD  
**Estado**: [Draft / En revisión / Aprobado]
