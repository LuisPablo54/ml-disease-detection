<div align="center">

# 🩸 Detección de Condiciones Hematológicas mediante Aprendizaje Automático:  Comparación entre Datos Sintéticos y Reales (NHANES 2017–2020)  
###    

[![Preprint](https://img.shields.io/badge/DOI-10.13140%2FRG.2.2.35072.29440-blue?style=flat-square)](https://doi.org/10.13140/RG.2.2.35072.29440)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Estado-Completado-success?style=flat-square)

*Proyecto de detección automática de condiciones hematológicas usando modelos de Machine Learning, comparando datos sintéticos y datos reales de NHANES 2017–2020.*

</div>

---

## 📋 Descripción

Este trabajo desarrolla, entrena y evalúa modelos de aprendizaje automático para la **detección temprana** de las siguientes condiciones:

- **Anemia**
- **Talasemia**
- **Trombocitopenia**
- **Diabetes**

Se utilizaron dos fuentes de datos:
- Dataset sintético (`Multiple Disease Prediction`)
- Datos reales del estudio epidemiológico **NHANES 2017–2020**

Se realizó un análisis completo que incluye validación clínica de umbrales diagnósticos, selección de características (RFE), análisis de componentes principales (PCA), estudio de balanceo de clases (SMOTE y ADASYN), optimización de hiperparámetros y análisis de **domain shift** entre datos sintéticos y reales.

> ⚠️ **Nota:** Los modelos son herramientas de apoyo experimental y **no sustituyen** el diagnóstico médico profesional.

---

## 📊 Datasets

### 1. Dataset Sintético
- Origen: `Multiple Disease Prediction`
- Etiquetas: Anemia, Diabetes, Talasemia, Trombocitopenia y Sano

### 2. Datos Reales - NHANES 2017–2020
- Más de 5,000 registros tras limpieza
- Variables hematológicas y bioquímicas (hemoglobina, plaquetas, MCV, MCH, glucosa, HbA1c, etc.)
- Etiquetado según criterios clínicos internacionales (OMS, ADA, etc.)

---

## 🔬 Metodología

### Etapas principales

1. **Validación clínica** de umbrales diagnósticos
2. **Selección de características** mediante Recursive Feature Elimination (RFE)
3. **Análisis exploratorio** con PCA
4. **Balanceo de clases**: SMOTE y ADASYN
5. **Entrenamiento y optimización** de tres modelos:
   - Decision Tree
   - Support Vector Machine (SVM)
   - Multi-Layer Perceptron (MLP)
6. **Interpretabilidad** con valores SHAP
7. **Clasificación multietiqueta** (Binary Relevance y Classifier Chains)
8. **Evaluación de domain shift** (sintético → reales)

---

## 🎯 Resultados Principales

- **Datos sintéticos**: Rendimiento cercano a la perfección (F1-Macro ≈ 1.00)
- **Transferencia a datos reales**: Severa degradación (hasta **75 puntos porcentuales** de pérdida)
- **Mejor configuración en datos reales**: **SVM + SMOTE** (mejor equilibrio entre Accuracy y F1-Macro)
- Confirmación de la importancia de entrenar y evaluar con datos reales de la población objetivo

---

## 🛠️ Tecnologías

- **Python** 3
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **SHAP** (interpretabilidad)
- **Imbalanced-learn** (SMOTE / ADASYN)

---

## ⚖️ Consideraciones Éticas

Este proyecto tiene fines **académicos y de investigación**. Los modelos desarrollados **no deben usarse** para diagnóstico clínico sin validación rigurosa y supervisión médica.

---

## 📄 Publicación

**Preprint publicado**:  
[10.13140/RG.2.2.14188.73608](https://doi.org/10.13140/RG.2.2.14188.73608)

**Artículo completo**: `Articulo_Final_DTE.pdf`

---

**Autores**:
- Paula Fernanda Rayas-López
- Luis Pablo López-Iracheta

---
