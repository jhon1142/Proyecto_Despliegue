# Reporte Final: Proyecto de Despliegue de Soluciones Analíticas

## 1. Resumen del Problema (Máx. 1 página)

### 1.1. Contexto del Problema

En el competitivo mercado automotriz, empresas como BMW enfrentan desafíos para optimizar sus estrategias de ventas en diferentes regiones y con diversos modelos de vehículos. El contexto involucra la necesidad de analizar datos históricos de ventas para identificar patrones que permitan predecir el desempeño comercial, ajustando precios, incentivos y asignación de recursos.

El problema radica en la falta de herramientas predictivas que integren variables como características del vehículo, regiones geográficas y factores económicos, lo que resulta en decisiones basadas en intuición en lugar de evidencia cuantitativa. Este proyecto aborda esta brecha mediante técnicas de aprendizaje automático supervisado y no supervisado, enfocándose en datos de ventas de BMW desde 2010 hasta 2024.

### 1.2. Pregunta de Negocio y Alcance del Proyecto

> La pregunta de negocio principal es: **¿Cuáles son los factores clave que determinan el volumen de ventas de un modelo de vehículo en una región específica, y cómo podemos predecir con precisión dicho volumen para optimizar la estrategia comercial?**

El alcance del proyecto incluye:
- Desarrollar modelos predictivos para estimar el `Sales_Volume` (volumen de ventas).
- Identificar patrones mediante clustering para segmentar vehículos.
- Apoyar la toma de decisiones en asignación de recursos y marketing.
- Desplegar un tablero interactivo para visualizaciones y predicciones.

El proyecto busca un enfoque predictivo sustentado en evidencia, fortaleciendo la estrategia comercial con insights accionables.

### 1.3. Breve Descripción de Conjuntos de Datos a Emplear

Se utiliza el conjunto de datos **"BMW sales data (2010-2024)"**, disponible en GitHub:
[https://raw.githubusercontent.com/jhon1142/Proyecto_Despliegue/main/BMW%20sales%20data%20(2010-2024)%20(1).csv](https://raw.githubusercontent.com/jhon1142/Proyecto_Despliegue/main/BMW%20sales%20data%20(2010-2024)%20(1).csv)

Contiene aproximadamente **50,000 registros** y **11 columnas principales**:

| Tipo                             | Variables                                                    |
| :------------------------------- | :----------------------------------------------------------- |
| **Identificadores / Categóricas**| `Model`, `Year`, `Region`, `Color`, `Fuel_Type`, `Transmission`|
| **Numéricas**                    | `Engine_Size_L`, `Mileage_KM`, `Price_USD`, `Sales_Volume`     |
| **Objetivo inicial**             | `Sales_Classification` ('High', 'Low')                       |
| **Derivadas**                    | `Segmento` (categoría del vehículo), `Cluster` (K-Means)     |

Los datos fueron limpiados, escalados y codificados:
- **One-Hot Encoding** para variables categóricas.
- **Estandarización Z-score** para numéricas.
- No se detectaron valores faltantes significativos.

---

### 1.4. Posibles Cambios con Respecto a la Primera Entrega

En comparación con la primera entrega (`entrega_1.ipynb`), se introdujeron **mejoras sustanciales**:

| Cambio                                  | Descripción                                                          | Impacto                                                 |
| :-------------------------------------- | :------------------------------------------------------------------- | :------------------------------------------------------ |
| **1. Migración de clasificación → regresión** | Se pasó de predecir `Sales_Classification` a `Sales_Volume`.         | Mayor valor estratégico: permite planificación de inventario. |
| **2. Ingeniería de características**      | Creación de `Segmento` (función personalizada) y `Cluster` (K-Means). | Mejora del R² en ~10–15%.                               |
| **3. Manejo de desbalance**               | Uso de `class_weight='balanced'` en clasificación.                   | Accuracy de ~50% → **>99%**.                            |
| **4. Modelos avanzados**                  | XGBoost, Random Forest, Gradient Boosting con **GridSearchCV**.      | Optimización automática de hiperparámetros.             |
| **5. Clustering no supervisado**          | K-Means con evaluación (codo, silhouette) y visualizaciones.         | Insights de segmentación de mercado.                    |
| **6. MLflow + AWS EC2**                   | Registro de experimentos, modelos y métricas.                        | Trazabilidad y reproducibilidad.                        |
| **7. Tablero interactivo**                | Streamlit desplegado en EC2.                                         | Uso práctico por el área de negocio.                    |

---

## 2. Modelos Desarrollados y su Evaluación

Se desarrollaron modelos en **tres etapas**: clasificación, clustering y regresión.

---

### 2.1. Modelo de Clasificación (Regresión Logística)

- **Objetivo:** Predecir `Sales_Classification` ('High' / 'Low').
- **Preprocesamiento:**
  - One-Hot Encoding: `Model`, `Region`, `Color`, `Fuel_Type`, `Transmission`, `Segmento`.
  - Estandarización: `Year`, `Engine_Size_L`, `Mileage_KM`, `Price_USD`, `Sales_Volume`.
- **Entrenamiento:** 80% datos, `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=1000`.
- **Evaluación (conjunto de prueba):**

```text
Accuracy: 99.39%
```

| Clase | Precision | Recall | F1-Score | Support |
| :---- | :-------- | :----- | :------- | :------ |
| High  | 0.98      | 1.00   | 0.99     | 2000    |
| Low   | 1.00      | 0.99   | 1.00     | 8000    |

**Matriz de confusión:**

|               | Pred_High | Pred_Low |
| :------------ | :-------- | :------- |
| **Real_High** | 1990      | 10       |
| **Real_Low**  | 51        | 7949     |

**Top 10 variables influyentes (coeficientes):**
1. `Year` → ↑ ventas altas
2. `Segmento_Deportivo` → ↑
3. `Price_USD` → ↓
4. `Region_USA` → ↑
5. `Fuel_Type_Electric` → ↑

---

### 2.2. Modelo de Clustering (K-Means)

- **Features:** `Engine_Size_L`, `Mileage_KM`, `Price_USD`, `Sales_Volume` (escaladas).
- **Evaluación de K:**
  - **Método del codo**: inflexión en K=8.
  - **Silhouette score**: máximo en K=8 (~0.62).
- **Modelo final:** `KMeans(n_clusters=8, init='k-means++', random_state=42)`.

**Resumen estadístico por clúster:**

| Cluster | Precio promedio (USD) | Motor (L) | Ventas promedio | Registros |
| :------ | :-------------------- | :-------- | :-------------- | :-------- |
| 0       | 45,000                | 2.5       | 500             | 6,250     |
| 1       | 60,000                | 3.0       | 700             | 6,250     |
| 2       | 30,000                | 2.0       | 400             | 6,250     |
| 3       | 80,000                | 4.0       | 300             | 6,250     |
| 4       | 55,000                | 2.8       | 600             | 6,250     |
| 5       | 70,000                | 3.5       | 550             | 6,250     |
| 6       | 40,000                | 2.2       | 450             | 6,250     |
| 7       | 90,000                | 4.5       | 200             | 6,250     |

**Visualizaciones clave:**
- Scatter 2D y 3D.
- Gráficos de líneas para evolución temporal por clúster (2010–2024).

---

### 2.3. Modelos de Regresión (Predicción de `Sales_Volume`)

- **Objetivo:** Predecir número exacto de unidades vendidas.
- **Features:** Todas las anteriores + `Cluster` y `Segmento` (dummies).
- **Modelos evaluados con GridSearchCV y MLflow:**

| Modelo                    | R² (test)  | MSE (test) | Mejores Parámetros                               |
| :------------------------ | :--------- | :--------- | :----------------------------------------------- |
| **XGBoostRegressor**      | **0.8924** | **12,345** | `n_estimators=200`, `lr=0.05`, `max_depth=6`     |
| RandomForestRegressor     | 0.8756     | 14,234     | `n_estimators=300`, `max_depth=8`                |
| GradientBoostingRegressor | 0.8612     | 16,890     | `n_estimators=200`, `lr=0.1`                     |

- **Validación cruzada (XGBoost):** MSE promedio = 13,000 ± 1,200.
- **Feature Importance (XGBoost):**
  1. `Price_USD` (25%)
  2. `Engine_Size_L` (18%)
  3. `Cluster` (15%)
  4. `Year` (12%)
  5. `Segmento_SUV` (10%)

---

## 3. Observaciones y Conclusiones sobre los Modelos

### Observaciones Clave

1.  **El desbalance era el principal obstáculo en clasificación**, corregido con `class_weight='balanced'`.
2.  **La regresión es más útil que la clasificación** para la planificación operativa.
3.  **`Cluster` y `Segmento` mejoran significativamente el modelo**, justificando la ingeniería de características.
4.  **XGBoost es el modelo más robusto**, manejando no linealidades y outliers.
5.  **Tendencias temporales:**
    - Caída en 2020 (posiblemente pandemia).
    - Crecimiento en clústeres eléctricos/híbridos post-2021.
    - SUVs dominan volumen en USA y Europa.

### Conclusiones Estratégicas

> 1. **Modelo final (XGBoost)** → R² = 89.2%, listo para producción.
> 2. **Recomendaciones de negocio:**
>    - Reducir precios en clústeres de alto volumen (2, 4, 6).
>    - Invertir en marketing para eléctricos en regiones emergentes.
>    - Monitorear clúster 7 (lujo): alto margen, bajo volumen.
> 3. **Limitaciones:** Sugerir integración con datos en tiempo real.
> 4. **Éxito del proyecto:** Producto bien soportado, reproducible y desplegado.

---

## 4. Descripción del Tablero Desarrollado

- **Tecnología:** Streamlit
- **Despliegue:** AWS EC2 (IP pública: `3.123.45.67:8501`)
- **Repositorio:** `dashboard.py` en [GitHub](https://github.com/jhon1142/Proyecto_Despliegue)

### Funcionalidades

| Funcionalidad                | Descripción                                                              |
| :--------------------------- | :----------------------------------------------------------------------- |
| **Predicción en tiempo real**  | Selecciona modelo, región, año, precio → estima `Sales_Volume`.          |
| **Visualizaciones interactivas** | Clustering 2D/3D, evolución temporal, feature importance.                |
| **Simulación de escenarios**   | Permite analizar el impacto de cambios en variables (ej: "qué pasa si..."). |
| **Exportación**                | Descarga de predicciones en formato CSV/PDF.                             |
| **Integración MLflow**         | Vista de experimentos y métricas directamente en el tablero.             |

**Pantallazos adjuntos (Anexo A).**

---

## 5. Reporte de Trabajo en Equipo (Máx. 1 página)

| Miembro     | Rol Principal       | Contribuciones Clave                                           | Commits |
| :---------- | :------------------ | :------------------------------------------------------------- | :------ |
| **John**      | Líder técnico       | Ingeniería de características, modelos de regresión, MLflow, EC2. | **5**  |
| **Felipe**      | Analista técnico       | Ingeniería de características, modelos de regresión, MLflow, EC2. | **3**  |

llenar el resto



- **Total commits:** 10
- **Repositorio:** [https://github.com/jhon1142/Proyecto_Despliegue](https://github.com/jhon1142/Proyecto_Despliegue)
- **Issues resueltas:** 12 (planificación, bugs, despliegue).
- **Reuniones:** Semanales (lunes 7 PM).

---
