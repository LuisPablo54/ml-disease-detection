# Early Disease Detection using Machine Learning

Este repositorio contiene el desarrollo de un proyecto de **aprendizaje automático aplicado a datos clínicos**, enfocado en la **detección temprana de posibles enfermedades** a partir de parámetros obtenidos de exámenes sanguíneos y variables clínicas básicas.

## Descripción del proyecto

El objetivo principal de este proyecto es analizar y modelar datos clínicos como niveles de glucosa, presión sanguínea, índice de masa corporal y otros indicadores relevantes, utilizando técnicas de **Machine Learning supervisado**, con el fin de identificar patrones asociados al riesgo de enfermedades comunes, particularmente metabólicas.

Los modelos desarrollados buscan funcionar como una **herramienta de apoyo al análisis clínico**, sin sustituir el diagnóstico médico profesional.

## Dataset

El conjunto de datos incluye las siguientes variables:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- **Outcome** (variable objetivo)

El dataset cuenta con 2000 registros y variables numéricas, adecuado para análisis exploratorio, modelado predictivo e interpretación clínica.

## Metodología

El proyecto sigue las siguientes etapas:

1. Análisis exploratorio de datos (EDA)
2. Limpieza y preprocesamiento
3. Análisis estadístico de variables clínicas
4. Entrenamiento de modelos de Machine Learning:
   - Regresión logística / múltiple
   - Árboles de decisión
5. Evaluación de modelos mediante métricas:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
6. Interpretación de resultados y análisis de errores

## Objetivo

- Identificar patrones clínicos asociados a posibles enfermedades
- Comparar el desempeño de distintos modelos de aprendizaje automático
- Evaluar la interpretabilidad de los modelos en un contexto clínico
- Analizar el potencial del Machine Learning para la detección temprana de riesgos en salud

## Consideraciones éticas

Este proyecto tiene fines **académicos y de investigación**.  
Los resultados obtenidos no deben considerarse diagnósticos médicos ni sustituir la evaluación de profesionales de la salud.

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

## Estructura del repositorio (sugerida)

