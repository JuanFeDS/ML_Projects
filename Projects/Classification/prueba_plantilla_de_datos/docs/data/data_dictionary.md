# Data Dictionary — Spaceship Titanic

Dataset de Kaggle para predecir si un pasajero fue transportado a otra dimensión
durante la colisión de la nave espacial Titanic con una anomalía espacio-temporal.

---

## Variables originales (raw)

| Variable | Tipo | Descripción | Valores / Rango |
|---|---|---|---|
| `PassengerId` | String (ID) | Identificador único del pasajero. Formato `GGGG_PP` donde `GGGG` es el grupo de viaje y `PP` la posición dentro del grupo. | Texto único |
| `HomePlanet` | Categórica | Planeta de origen del pasajero. | `Earth`, `Europa`, `Mars` |
| `CryoSleep` | Booleana | Indica si el pasajero optó por suspensión criogénica durante el viaje. Los pasajeros en CryoSleep no pueden gastar. | `True`, `False` |
| `Cabin` | String | Cabina asignada al pasajero. Formato `Deck/CabinNumber/Side`. | e.g. `B/0/P` |
| `Destination` | Categórica | Destino final del pasajero. | `TRAPPIST-1e`, `PSO J318.5-22`, `55 Cancri e` |
| `Age` | Numérica | Edad del pasajero en años. | 0 – 79 |
| `VIP` | Booleana | Si el pasajero contrató servicio VIP. | `True`, `False` |
| `RoomService` | Numérica | Gasto en servicio de habitación (créditos). | 0 – 14,327 |
| `FoodCourt` | Numérica | Gasto en cafetería (créditos). | 0 – 29,813 |
| `ShoppingMall` | Numérica | Gasto en tiendas (créditos). | 0 – 23,492 |
| `Spa` | Numérica | Gasto en spa (créditos). | 0 – 22,408 |
| `VRDeck` | Numérica | Gasto en sala de realidad virtual (créditos). | 0 – 24,133 |
| `Name` | String | Nombre completo del pasajero. | Texto libre |
| `Transported` | Booleana | **TARGET** — Si el pasajero fue transportado a otra dimensión. | `True`, `False` |

---

## Variables derivadas (feature engineering)

| Variable | Origen | Descripción | Acción |
|---|---|---|---|
| `Deck` | `Cabin` | Cubierta de la nave (letra del formato `Deck/Number/Side`). Alta discriminación: chi²≈392. | MANTENER |
| `CabinNumber` | `Cabin` | Número de cabina dentro del deck. Señal de posición a bordo. | MANTENER |
| `Side` | `Cabin` | Lado del barco (`P`=Port, `S`=Starboard). Señal cubierta por Deck. | DESCARTAR |
| `TotalSpending` | Suma de gastos | Suma bruta de RoomService + FoodCourt + ShoppingMall + Spa + VRDeck. | DESCARTAR (reemplazada por log) |
| `TotalSpending_Log` | `TotalSpending` | log1p(TotalSpending). Mejora correlación con target de r≈-0.32 a r≈-0.42. | MANTENER |
| `GroupSize` | `PassengerId` | Número de pasajeros que viajan juntos en el mismo grupo. chi²>0, p<0.001. | MANTENER |
| `SpendingCategories` | Gastos | Número de categorías de gasto con valor > 0 (0–5). | MANTENER |
| `AgeCategory` | `Age` | Categorización por tramos de edad (bins definidos en `constants.py`). | MANTENER |

---

## Notas de preprocesamiento

- **Nulos en categóricas** (`HomePlanet`, `Destination`, `CryoSleep`, `Deck`, `Side`): imputados con la moda o con categoría `'Unknown'`.
- **Nulos en numéricas** (`Age`, gastos): imputados con 0 (gastos) o mediana (Age).
- **Encoding**: variables categóricas codificadas con Label Encoding o One-Hot según el modelo.
- **Escalado**: features numéricas escaladas con `StandardScaler` (guardado en `models/production/scaler.pkl`).
- `PassengerId`, `Name` y `Side` descartados del modelo final.
