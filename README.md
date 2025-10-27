# Lab 11 – Proyecto de Consultoría: **Churn** con PySpark
Gabriel Paz
Carlos Valladares

> README rápido para correr y entregar el laboratorio con **PySpark** usando Regresión Logística.

## Objetivo
Entrenar un modelo de **clasificación** (Regresión Logística con `pyspark.ml`) que prediga **churn** usando datos históricos y aplicar el modelo a **clientes nuevos** para priorizar la asignación de gerentes.

## Archivos
- `abandono_clientes.csv` → histórico con la etiqueta `Churn` (0/1).
- `clientes_nuevos.csv` → clientes futuros (sin etiqueta para inferir).
- `Lab11_RLog_Spark.ipynb` o `Lab11_RLog_Spark_v2.ipynb` → notebook PySpark con el pipeline y evaluación.
- `predicciones_clientes_nuevos.csv` → salida con `Churn_pred`, `Churn_proba`, `Asignar_Gerente`.

## Requisitos
- Python 3.10+  
- Paquetes:
  ```bash
  pip install pyspark matplotlib
  ```
- Jupyter o VS Code con Jupyter.

> Si corres en Colab, primero: `!pip install pyspark matplotlib`.

## Cómo correr
1. Colocar `abandono_clientes.csv` y `clientes_nuevos.csv` en la **misma carpeta** del notebook.
2. Abrir `Lab11_RLog_Spark.ipynb` (o la versión v2).
3. Ejecutar **todas** las celdas en orden.  
   - Se entrena el modelo con un pipeline:  
     `Imputer + StringIndexer + OneHotEncoder + VectorAssembler + StandardScaler + LogisticRegression`.
   - Se evalúa con: **accuracy, f1 (weighted), ROC-AUC, AUPR**, matriz de confusión y **curva ROC**.
   - Se generan predicciones para clientes nuevos y se guarda `predicciones_clientes_nuevos.csv`.

## Qué entrega el notebook
- **Métricas** en test (ejemplo real obtenido):
  - Accuracy ≈ **0.8021**
  - F1 (weighted) ≈ **0.7192**
  - ROC-AUC ≈ **0.89**
  - AUPR ≈ **0.746**
- **Matriz de confusión** (umbral 0.5) y **curva ROC**.
- CSV final con las columnas:
  - `Churn_pred` (0/1), `Churn_proba` (prob. clase 1),
  - `Asignar_Gerente` (regla simple basada en umbral).

## Regla de Negocio sugerida
- Marcar `Asignar_Gerente = 1` cuando `Churn_proba >= 0.5` (o el umbral que decida el equipo).
- Si se prioriza **recall** (captar más abandonos), bajar umbral a **0.35–0.40**.

### Cambiar el umbral (rápido)
En la celda de predicción a clientes nuevos, puedes ajustar la regla:
```python
umbral = 0.40  # por ejemplo
out = out.withColumn("Asignar_Gerente", (F.col("Churn_proba") >= F.lit(umbral)).cast("int"))
```


## Problemas comunes y fixes
- **Fechas** con formatos raros: el notebook intenta varios formatos y crea `*_days_since`.  
- **AUC/ROC**: si tu entorno no trae utilidades antiguas, la **curva ROC** se calcula en pandas/numpy (compatible).
- **Faltan columnas en predicción**: asegúrate de correr la celda de **preparación de features** para `df_new` antes de predecir.

## Qué entregar
- Notebook ejecutado con las celdas completas (pipeline, métricas, matriz, ROC).
- `predicciones_clientes_nuevos.csv` con las 3 columnas: `Churn_pred`, `Churn_proba`, `Asignar_Gerente`.
- Este `README.md`.
