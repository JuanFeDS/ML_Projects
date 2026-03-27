# EDA — Spaceship Titanic


## Analisis Basico del Dataset

- **Filas:** 8,693

- **Columnas:** 14


### Tipos de datos

| Columna      | Tipo    |   Valores unicos |
|:-------------|:--------|-----------------:|
| PassengerId  | object  |             8693 |
| HomePlanet   | object  |                3 |
| CryoSleep    | object  |                2 |
| Cabin        | object  |             6560 |
| Destination  | object  |                3 |
| Age          | float64 |               80 |
| VIP          | object  |                2 |
| RoomService  | float64 |             1273 |
| FoodCourt    | float64 |             1507 |
| ShoppingMall | float64 |             1115 |
| Spa          | float64 |             1327 |
| VRDeck       | float64 |             1306 |
| Name         | object  |             8473 |
| Transported  | bool    |                2 |


### Valores nulos

| Columna      |   Nulos |   % Nulos |
|:-------------|--------:|----------:|
| HomePlanet   |     201 |      2.31 |
| CryoSleep    |     217 |      2.5  |
| Cabin        |     199 |      2.29 |
| Destination  |     182 |      2.09 |
| Age          |     179 |      2.06 |
| VIP          |     203 |      2.34 |
| RoomService  |     181 |      2.08 |
| FoodCourt    |     183 |      2.11 |
| ShoppingMall |     208 |      2.39 |
| Spa          |     183 |      2.11 |
| VRDeck       |     188 |      2.16 |
| Name         |     200 |      2.3  |

- **Duplicados exactos:** 0 (0.00%)


## Balance del Target (Transported)

| Clase   |   Conteo |   % del total |
|:--------|---------:|--------------:|
| True    |     4378 |         50.36 |
| False   |     4315 |         49.64 |


## Variables Numericas


### Estadisticas descriptivas

|              |   count |   mean |     std |   min |   25% |   50% |   75% |   max |
|:-------------|--------:|-------:|--------:|------:|------:|------:|------:|------:|
| Age          |    8514 |  28.83 |   14.49 |     0 |    19 |    27 |    38 |    79 |
| RoomService  |    8512 | 224.69 |  666.72 |     0 |     0 |     0 |    47 | 14327 |
| FoodCourt    |    8510 | 458.08 | 1611.49 |     0 |     0 |     0 |    76 | 29813 |
| ShoppingMall |    8485 | 173.73 |  604.7  |     0 |     0 |     0 |    27 | 23492 |
| Spa          |    8510 | 311.14 | 1136.71 |     0 |     0 |     0 |    59 | 22408 |
| VRDeck       |    8505 | 304.85 | 1145.72 |     0 |     0 |     0 |    46 | 24133 |


### Correlaciones con el target

| Feature      |   Pearson r |
|:-------------|------------:|
| RoomService  |      -0.245 |
| Spa          |      -0.221 |
| VRDeck       |      -0.207 |
| Age          |      -0.075 |
| FoodCourt    |       0.047 |
| ShoppingMall |       0.01  |


## Variables Categoricas


### HomePlanet

- **chi2:** 324.97

- **p-valor:** 3.92e-70

- **Grados de libertad:** 3

| HomePlanet   |   count |   pct |   tasa_transported |
|:-------------|--------:|------:|-------------------:|
| Earth        |    4602 | 52.94 |             0.4239 |
| Europa       |    2131 | 24.51 |             0.6588 |
| Mars         |    1759 | 20.23 |             0.523  |
| nan          |     201 |  2.31 |             0.5124 |


### CryoSleep

- **chi2:** 1861.75

- **p-valor:** 0.00e+00

- **Grados de libertad:** 2

|   CryoSleep |   count |   pct |   tasa_transported |
|------------:|--------:|------:|-------------------:|
|           0 |    5439 | 62.57 |             0.3289 |
|           1 |    3037 | 34.94 |             0.8176 |
|         nan |     217 |  2.5  |             0.4885 |


### Destination

- **chi2:** 106.39

- **p-valor:** 6.55e-23

- **Grados de libertad:** 3

| Destination   |   count |   pct |   tasa_transported |
|:--------------|--------:|------:|-------------------:|
| TRAPPIST-1e   |    5915 | 68.04 |             0.4712 |
| 55 Cancri e   |    1800 | 20.71 |             0.61   |
| PSO J318.5-22 |     796 |  9.16 |             0.5038 |
| nan           |     182 |  2.09 |             0.5055 |


### VIP

- **chi2:** 12.1

- **p-valor:** 2.36e-03

- **Grados de libertad:** 2

|   VIP |   count |   pct |   tasa_transported |
|------:|--------:|------:|-------------------:|
|     0 |    8291 | 95.38 |             0.5063 |
|   nan |     203 |  2.34 |             0.5123 |
|     1 |     199 |  2.29 |             0.3819 |


## Variables Numericas vs Target

| Feature      |   MannWhitney stat |   MW p-valor |   Pearson r |   Media (True) |   Media (False) |
|:-------------|-------------------:|-------------:|------------:|---------------:|----------------:|
| Age          |        8.31763e+06 |    5.62e-11  |     -0.075  |         27.749 |          29.923 |
| RoomService  |        5.7658e+06  |    6.65e-257 |     -0.2446 |         63.098 |         389.266 |
| FoodCourt    |        7.37345e+06 |    8.88e-67  |      0.0466 |        532.692 |         382.616 |
| ShoppingMall |        6.99159e+06 |    2.19e-98  |      0.0101 |        179.83  |         167.566 |
| Spa          |        5.66778e+06 |    7.45e-259 |     -0.2211 |         61.676 |         564.383 |
| VRDeck       |        5.89819e+06 |    1.55e-231 |     -0.2071 |         69.148 |         543.63  |


## Features Derivadas


### TotalSpending vs TotalSpending_Log

- **r(TotalSpending, Target):** -0.1995

- **r(TotalSpending_Log, Target):** -0.4689

La transformacion log mejora la correlacion absoluta de 0.200 a 0.469.


### GroupSize

- **chi2:** 145.28

- **p-valor:** 3.97e-28

|   GroupSize |   tasa_transported |    n |
|------------:|-------------------:|-----:|
|           1 |           0.452445 | 4805 |
|           2 |           0.53805  | 1682 |
|           3 |           0.593137 | 1020 |
|           4 |           0.640777 |  412 |
|           5 |           0.592453 |  265 |
|           6 |           0.614943 |  174 |
|           7 |           0.541126 |  231 |
|           8 |           0.394231 |  104 |


### SpendingCategories

- **chi2:** 2024.08

- **p-valor:** 0.00e+00

|   SpendingCategories |   tasa_transported |    n |
|---------------------:|-------------------:|-----:|
|                    0 |           0.786477 | 3653 |
|                    1 |           0.346863 |  271 |
|                    2 |           0.299837 | 1224 |
|                    3 |           0.306576 | 2068 |
|                    4 |           0.269388 | 1225 |
|                    5 |           0.31746  |  252 |


## Tabla de Decisiones

| Feature            | Accion              | Tipo         | Justificacion                                |
|:-------------------|:--------------------|:-------------|:---------------------------------------------|
| Age                | MANTENER            | Numerica     | MW p<0.001; categorizar en AgeCategory       |
| RoomService        | MANTENER            | Numerica     | Correlacion con target via TotalSpending_Log |
| FoodCourt          | MANTENER            | Numerica     | Idem RoomService; parte de TotalSpending     |
| ShoppingMall       | MANTENER            | Numerica     | Idem                                         |
| Spa                | MANTENER            | Numerica     | Idem                                         |
| VRDeck             | MANTENER            | Numerica     | Idem                                         |
| TotalSpending_Log  | MANTENER (derivada) | Numerica     | r=-0.469 vs r=-0.2 crudo                     |
| TotalSpending      | DESCARTAR           | Numerica     | Reemplazada por TotalSpending_Log            |
| GroupSize          | MANTENER (derivada) | Numerica/Cat | chi2=145.3, p<0.001                          |
| SpendingCategories | MANTENER (derivada) | Numerica     | chi2=2024.1, p<0.001                         |
| Cabin              | TRANSFORMAR         | Categorica   | Extraer Deck (chi2~392), CabinNumber, Side   |
| Deck               | MANTENER (derivada) | Categorica   | Alta discriminacion, chi2>392                |
| Side               | DESCARTAR           | Categorica   | Senal cubierta por Deck                      |
| CabinNumber        | MANTENER (derivada) | Numerica     | Senal de posicion a bordo                    |
| HomePlanet         | MANTENER            | Categorica   | chi2 significativo                           |
| Destination        | MANTENER            | Categorica   | chi2 significativo                           |
| CryoSleep          | MANTENER (encoded)  | Binaria      | Alta correlacion con no-gasto; label encode  |
| VIP                | DESCARTAR           | Binaria      | Senal debil (corr~-0.037)                    |
| Name               | DESCARTAR           | Texto        | Sin poder predictivo directo                 |
| PassengerId        | DESCARTAR           | ID           | Solo ID; info util ya extraida en GroupSize  |
