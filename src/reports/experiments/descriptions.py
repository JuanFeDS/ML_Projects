"""
Descripciones narrativas de cada experimento de feature engineering.

Editar este archivo para actualizar el contenido de las tarjetas de detalle
en el reporte HTML de seguimiento de experimentos.

Estructura de cada entrada:
    "fs-NNN_nombre": {
        "what":       str  — que features se agregaron/cambiaron,
        "hypothesis": str  — por que se esperaba que mejorara,
        "result":     str  — que paso realmente,
        "tags":       list — etiquetas de contexto (opcionales),
    }
"""

EXPERIMENT_DESCRIPTIONS: dict = {
    "fs-001_baseline": {
        "what": (
            "Feature set base. Descomposicion de Cabin en Deck, CabinNum y Side. "
            "Extraccion de TravelGroup e IsAlone desde PassengerId. "
            "Creacion de TotalSpending y log1p(TotalSpending). "
            "Imputacion con medianas/modas. Target encoding para HomePlanet, Destination y Deck."
        ),
        "hypothesis": (
            "Establecer un pipeline completo end-to-end como punto de referencia. "
            "Las variables derivadas de Cabin y grupo de viaje tienen alta senhal estadistica "
            "(chi2 Deck=392, Side=91) y se esperaba una mejora directa sobre features crudas."
        ),
        "result": (
            "Val accuracy 0.8285 con CatBoost tuneado. Buen punto de partida. "
            "Confirma que Cabin y TotalSpending son las transformaciones mas valiosas del baseline."
        ),
        "tags": ["baseline", "cabin", "spending", "target-encoding"],
    },

    "fs-002_cryo_interactions": {
        "what": (
            "Agrega interacciones entre CryoSleep y las variables de gasto: "
            "CryoSleep_x_TotalSpending, CryoSleep_x_RoomService, etc. "
            "Tambien CryoSleep codificado como entero (0/1)."
        ),
        "hypothesis": (
            "CryoSleep es la variable mas discriminativa (chi2=1861). "
            "Multiplicarla por gasto deberia amplificar la senhal para modelos lineales "
            "que no capturan interacciones automaticamente."
        ),
        "result": (
            "Val accuracy 0.7868 — caida respecto al baseline. Las interacciones "
            "introducen multicolinealidad sin aportar senhal adicional a modelos de arboles, "
            "que ya capturan estas interacciones internamente."
        ),
        "tags": ["interactions", "cryo", "spending"],
    },

    "fs-003_solo_interactions": {
        "what": (
            "Agrega features de viajero solo: IsAlone_x_TotalSpending, "
            "IsAlone_x_CryoSleep. Combina el estado de aislamiento social "
            "con las variables de mayor poder predictivo."
        ),
        "hypothesis": (
            "Los viajeros solos podrian tener patrones de comportamiento distintos "
            "en gasto y suenho criogenico. La interaccion deberia capturar este subgrupo."
        ),
        "result": (
            "Val accuracy 0.7892. Ligera mejora sobre fs-002 pero sigue por debajo del baseline. "
            "Los arboles de decision ya aprenden estas particiones sin necesitar features explicitas."
        ),
        "tags": ["interactions", "solo", "group"],
    },

    "fs-004_target_encoding": {
        "what": (
            "Refinamiento del target encoding. Aplica TE con suavizado bayesiano (smoothing=10) "
            "a HomePlanet, Destination, Deck y Side. "
            "Elimina las interacciones redundantes de fs-002/003."
        ),
        "hypothesis": (
            "El target encoding con suavizado evita sobreajuste en categorias con pocos ejemplos. "
            "Encodear Side (chi2=91) captura la asimetria P/S que OHE pierde."
        ),
        "result": (
            "Val accuracy 0.8238 con 50 trials Optuna. Recupera el nivel del baseline "
            "con un pipeline mas limpio. Se convierte en el fs de referencia para experimentos "
            "de modelos (exp-010 a exp-015)."
        ),
        "tags": ["target-encoding", "smoothing", "cabin"],
    },

    "fs-005_structural_context": {
        "what": (
            "Agrega features de contexto estructural del grupo de viaje: "
            "GroupSize, GroupCryoRatio (proporcion de miembros en cryo), "
            "GroupSpendingMean y GroupSpendingStd."
        ),
        "hypothesis": (
            "El grupo de viaje comparte caracteristicas fisicas (mismo Deck/Side). "
            "El ratio de CryoSleep dentro del grupo puede predecir el comportamiento "
            "de miembros cuyo estado CryoSleep es nulo."
        ),
        "result": (
            "Val accuracy 0.8233 con 25 trials. Marginalmente por debajo de fs-004. "
            "El GroupCryoRatio tiene senhal pero el ruido de grupos pequenos la diluye."
        ),
        "tags": ["group", "context", "aggregation"],
    },

    "fs-006_group_imputation": {
        "what": (
            "Imputacion de nulos guiada por el grupo de viaje: si todos los demas miembros "
            "del grupo tienen CryoSleep=True, se imputa True al miembro con NaN. "
            "Idem para HomePlanet y Cabin."
        ),
        "hypothesis": (
            "Las reglas de dominio indican que el grupo comparte Cabin y probablemente "
            "CryoSleep. Resolver NaN por inferencia antes de imputacion estadistica "
            "deberia preservar la distribucion real."
        ),
        "result": (
            "Val accuracy 0.7874. La imputacion por grupo introduce sesgo cuando el grupo "
            "tiene comportamiento mixto. Los nulos en CryoSleep son pocos (~2%) "
            "y la ganancia no compensa el ruido."
        ),
        "tags": ["imputation", "group", "domain-rules"],
    },

    "fs-007_domain_rules": {
        "what": (
            "Aplica las 5 reglas de dominio fisicas del dataset como features directas: "
            "CryoSleep_Spending_Violation, Age_Spending_Violation, Deck_HomePlanet_Match, etc. "
            "Las violaciones se codifican como flags binarios."
        ),
        "hypothesis": (
            "Las reglas fisicas son casi perfectas (<0.1% violaciones). "
            "Codificarlas explicitamente deberia dar al modelo informacion de alta confianza "
            "sobre la consistencia interna de cada pasajero."
        ),
        "result": (
            "Val accuracy 0.9454 — anomalamente alta. Las reglas de dominio "
            "codifican el target casi directamente (CryoSleep=True implica gasto=0 "
            "implica Transported=True con >80% de precision). "
            "La submission confirma data leakage: tasa de transporte 50.2%, la mas baja."
        ),
        "tags": ["domain-rules", "data-leakage", "anomaly"],
    },

    "fs-008_domain_rules_only": {
        "what": (
            "Version depurada de fs-007. Solo usa las reglas de dominio que NO derivan "
            "directamente del target: Deck_HomePlanet_Match y Age_Spending_Flag. "
            "Elimina las reglas de CryoSleep."
        ),
        "hypothesis": (
            "Las reglas de HomePlanet×Deck y Age×Spending son fisicamente validas "
            "y no representan fuga. Deberian aportar senhal sin leakage."
        ),
        "result": (
            "Val accuracy 0.7898. Sin la regla de CryoSleep la mejora desaparece. "
            "Confirma que la ganancia de fs-007 era enteramente por leakage."
        ),
        "tags": ["domain-rules", "ablation"],
    },

    "fs-009_percentile_cabin": {
        "what": (
            "Agrega CabinNum_Percentile: la posicion percentil del numero de cabina "
            "dentro de cada Deck. Hipotesis: las cabinas mas altas o bajas en cada deck "
            "pueden corresponder a distintas zonas socioeconomicas."
        ),
        "hypothesis": (
            "El numero de cabina dentro de un deck puede capturar jerarquia espacial "
            "que el deck solo no expresa. Un percentil es invariante a la escala del deck."
        ),
        "result": (
            "Val accuracy 0.7886. Sin mejora apreciable. El numero de cabina dentro "
            "del deck no tiene poder predictivo adicional mas alla del Deck mismo."
        ),
        "tags": ["cabin", "percentile", "spatial"],
    },

    "fs-010_cryo_spending": {
        "what": (
            "Agrega features de interaccion CryoSleep × cada servicio individual: "
            "CryoSleep_x_RoomService, CryoSleep_x_FoodCourt, CryoSleep_x_ShoppingMall, "
            "CryoSleep_x_Spa, CryoSleep_x_VRDeck. "
            "Tambien SpendingEntropy (diversidad de servicios usados)."
        ),
        "hypothesis": (
            "La zero-inflation en los servicios es extrema para pasajeros en CryoSleep. "
            "Separar el patron de gasto por estado criogenico puede ayudar al modelo "
            "a aprender la distribucion bimodal."
        ),
        "result": (
            "Val accuracy 0.8150 con 25 trials. Mejora sobre los experimentos de interacciones "
            "anteriores pero no supera fs-004. SpendingEntropy no aporta senhal significativa."
        ),
        "tags": ["cryo", "spending", "interactions", "zero-inflation"],
    },

    "fs-011_child_route": {
        "what": (
            "Agrega features especificas para ninos (Age <= 12): IsChild flag, "
            "ChildOnCryo (nino en suenho criogenico), ChildWithSpending (nino con gasto > 0). "
            "Tambien RouteTE: target encoding de la ruta HomePlanet → Destination."
        ),
        "hypothesis": (
            "Los ninos tienen restricciones en CryoSleep y gasto. "
            "La ruta de viaje combina origen y destino en una sola feature que puede "
            "capturar patrones demograficos de la travesia."
        ),
        "result": (
            "Val accuracy 0.8180 con 25 trials. Mejora real sobre fs-010. "
            "IsChild y RouteTE contribuyen. ChildWithSpending tiene baja frecuencia "
            "pero alta presicion predictiva."
        ),
        "tags": ["age", "children", "route", "target-encoding"],
    },

    "fs-012_child_route_te": {
        "what": (
            "Extiende fs-011 con target encoding completo para la ruta "
            "(HomePlanet_Destination_TE) y agrega GroupHomePlanet_TE "
            "(TE del HomePlanet mayoritario dentro del grupo de viaje). "
            "Modelo ganador: CatBoost con 400 iteraciones."
        ),
        "hypothesis": (
            "Combinar el contexto de ruta con el contexto grupal deberia capturar "
            "la interaccion entre origen del grupo y comportamiento colectivo. "
            "El TE del grupo reduce la cardinalidad sin perder senhal."
        ),
        "result": (
            "Val accuracy 0.8250 — mejor resultado real del proyecto (exp-017). "
            "Promovido a produccion. El GroupHomePlanet_TE es la feature nueva "
            "con mayor importancia SHAP en este experimento."
        ),
        "tags": ["target-encoding", "route", "group", "production", "best"],
    },

    "fs-013_group_context": {
        "what": (
            "Agrega features de contexto de grupo orientadas a CryoSleep colectivo: "
            "GroupAllCryo (todos en cryo), GroupAnyCryo (alguno en cryo), "
            "GroupCryoConsensus (acuerdo interno). "
            "Motivadas por el heatmap HomePlanet x CryoSleep del EDA."
        ),
        "hypothesis": (
            "Si todos los miembros del grupo estan en CryoSleep, el pasajero con NaN "
            "probablemente tambien lo esta. Codificar el consenso grupal como feature "
            "explicita ayuda mas que la imputacion implicita."
        ),
        "result": (
            "Val accuracy 0.8203 con 50 trials. Por debajo de fs-012. "
            "El consenso grupal aporta senhal pero la interaccion con el target encoding "
            "de ruta de fs-012 es mas potente. Modelo ganador: MoE (Mixture of Experts)."
        ),
        "tags": ["group", "cryo", "context", "consensus", "moe"],
    },
}
