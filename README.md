<div align="center">

# 🩸 Detección Temprana de Enfermedades Hematológicas
### mediante Aprendizaje Automático con Datos de Análisis Sanguíneos

[![Preprint](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.35072.29440-blue?style=flat-square)](https://doi.org/10.13140/RG.2.2.35072.29440)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Estado-En%20desarrollo-yellow?style=flat-square)

*Proyecto de aprendizaje automático aplicado a datos clínicos para la detección temprana de enfermedades, validado con datos reales de la* ***NHANES***

</div>

---

## 📋 Descripción

El objetivo principal es analizar y modelar parámetros clínicos —niveles de glucosa, presión sanguínea, IMC y otros indicadores— usando técnicas de **Machine Learning supervisado**, con el fin de identificar patrones asociados al riesgo de enfermedades metabólicas.

> ⚠️ **Nota:** Los modelos desarrollados funcionan como **herramienta de apoyo al análisis clínico** y no sustituyen el diagnóstico médico profesional.

---

## 📊 Dataset

El conjunto de datos de entrenamiento cuenta con **2,000 registros** y las siguientes variables:

| Variable | Descripción |
|---|---|
| `Pregnancies` | Número de embarazos |
| `Glucose` | Nivel de glucosa en sangre |
| `BloodPressure` | Presión arterial |
| `SkinThickness` | Grosor del pliegue cutáneo |
| `Insulin` | Nivel de insulina |
| `BMI` | Índice de masa corporal |
| `DiabetesPedigreeFunction` | Función de pedigrí de diabetes |
| `Age` | Edad del paciente |
| `Outcome` | ⭐ Variable objetivo |

---

## 🔬 Metodología

### Etapa 1 — Modelado con datos simulados

**EDA → Preprocesamiento → Análisis estadístico → Entrenamiento → Evaluación**
1. **Análisis exploratorio de datos (EDA)**
2. **Limpieza y preprocesamiento**
3. **Análisis estadístico** de variables clínicas
4. **Entrenamiento de modelos:**
   - Regresión logística / múltiple
   - Árboles de decisión
5. **Evaluación mediante métricas:** Accuracy · Precision · Recall · F1-score · ROC-AUC
6. **Interpretación de resultados** y análisis de errores

---

### Etapa 2 — Validación con datos reales (NHANES)

Datos prepandemia **2017 – marzo 2020**, integrando los siguientes archivos:

| Archivo | Contenido |
|---|---|
| `P_DEMO.xpt` | Demografía (edad, sexo, etc.) |
| `P_CBC.xpt` | Hemograma completo (plaquetas, leucocitos, eritrocitos) |
| `P_BIOPRO.xpt` | Bioquímica (ALT, AST, Creatinina, Glucosa, Colesterol) |
| `P_TCHOL.xpt` | Colesterol total |
| `P_HDL.xpt` | HDL |
| `P_TRIGLY.xpt` | Triglicéridos + LDL |
| `P_GHB.xpt` | HbA1c |
| `P_INS.xpt` | Insulina |
| `P_HSCRP.xpt` | Proteína C Reactiva |
| `P_GLU.xpt` | Glucosa en ayunas |
| `P_BMX.xpt` | BMI, peso, talla |
| `P_BPXO.xpt` | Presión arterial + frecuencia cardíaca |

**Pasos planeados:**

2. Consolidación de archivos en un único dataset integrado
3. Etiquetado clínico por enfermedad para validación rudimentaria de modelos
4. Prototipo de interfaz web que simule resultados de laboratorio médico

---

## 🎯 Objetivos

- [ ] Identificar patrones clínicos asociados a posibles enfermedades
- [ ] Comparar el desempeño de distintos modelos de ML
- [ ] Evaluar la interpretabilidad de los modelos en contexto clínico
- [ ] Analizar el potencial del ML para detección temprana de riesgos en salud

---

## 🛠️ Tecnologías

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## ⚖️ Consideraciones éticas

Este proyecto tiene fines **académicos y de investigación**. Los resultados obtenidos **no deben considerarse diagnósticos médicos** ni sustituir la evaluación de profesionales de la salud.

---

## 📄 Publicación

> Preprint publicado: **[10.13140/RG.2.2.35072.29440](https://doi.org/10.13140/RG.2.2.35072.29440)**

