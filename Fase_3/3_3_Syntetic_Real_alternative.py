# -*- coding: utf-8 -*-
"""
Fase 3.3 — Domain Shift con Múltiples Modelos y Normalización de NHANES

- Normaliza NHANES usando rangos fisiológicos para mapear a [0,1].
- Los datos sintéticos ya están en [0,1].
- Modelos: DecisionTree (max_depth=8), SVM (C=1000, γ=0.01), MLP (128-64-32)
- Escenarios: Sintético → Sintético (incluye diabetes), Sintético → NHANES (clases comunes)
- Balanceo: class_weight='balanced' para DT y SVM; MLP sin balanceo.
- División: 70% entrenamiento, 20% prueba, 10% validación (para early stopping)
- Métricas: Accuracy y F1-macro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =============================================================================
# 1. Cargar datos
# =============================================================================
df_sint = pd.read_csv("data/Blood_samples_dataset_balanced_2(f).csv")
df_nhanes = pd.read_csv("data_test/NHANES_2017_2020_labeled_diseases.csv")
print(f"Sintético shape : {df_sint.shape}")
print(f"NHANES shape    : {df_nhanes.shape}")

# =============================================================================
# 2. Normalizar NHANES con rangos fisiológicos (solo columnas presentes)
# =============================================================================
ranges = {
    "Platelets": (150, 450),
    "White Blood Cells": (4, 11),
    "Red Blood Cells": (4.2, 5.4),
    "Hematocrit": (38, 52),
    "Mean Corpuscular Volume": (80, 100),
    "Mean Corpuscular Hemoglobin": (27, 33),
    "Mean Corpuscular Hemoglobin Concentration": (32, 36),
    "HDL Cholesterol": (40, 60),
    "ALT": (10, 40),
    "Heart Rate": (60, 100)
}

df_nhanes_norm = df_nhanes.copy()
for col, (min_val, max_val) in ranges.items():
    if col in df_nhanes_norm.columns:
        df_nhanes_norm[col] = pd.to_numeric(df_nhanes_norm[col], errors='coerce')
        df_nhanes_norm[col] = (df_nhanes_norm[col] - min_val) / (max_val - min_val)
        df_nhanes_norm[col] = df_nhanes_norm[col].clip(0, 1)
    else:
        print(f"Advertencia: {col} no está en NHANES, se omite.")

print("\n--- Normalización de NHANES (primeras 5 filas de columnas numéricas) ---")
print(df_nhanes_norm[ranges.keys()].head())

# =============================================================================
# 3. Alinear columnas comunes
# =============================================================================
cols_nhanes = set(df_nhanes_norm.columns) - {'condition'}
cols_sint   = set(df_sint.columns) - {'Disease'}
cols_comunes = sorted(list(cols_nhanes & cols_sint))
print(f"\nColumnas compartidas: {len(cols_comunes)} → {cols_comunes}")

# Datos con solo columnas comunes
X_nhanes = df_nhanes_norm[cols_comunes].copy()
X_sint = df_sint[cols_comunes].copy()

# =============================================================================
# 4. Mapeo de etiquetas
# =============================================================================
# Etiquetas completas para sintético (incluye diabetes)
mapa_sint_full = {
    'Anemia': 'anemia', 'Healthy': 'healthy', 'Diabetes': 'diabetes',
    'Thalasse': 'thalassemia', 'Thromboc': 'thrombocytopenia'
}
df_sint['label_full'] = df_sint['Disease'].map(mapa_sint_full)

# Etiquetas comunes (sin diabetes ni borderline)
mapa_nhanes = {
    'anemia': 'anemia', 'healthy': 'healthy', 'borderline': 'other',
    'thalassemia': 'thalassemia', 'thrombocytopenia': 'thrombocytopenia'
}
df_nhanes_norm['label_common'] = df_nhanes_norm['condition'].map(mapa_nhanes)
df_nhanes_common = df_nhanes_norm[df_nhanes_norm['label_common'] != 'other'].copy()

clases_comunes = ['anemia', 'healthy', 'thalassemia', 'thrombocytopenia']
df_sint_common = df_sint[df_sint['label_full'].isin(clases_comunes)].copy()
df_sint_common['label_common'] = df_sint_common['label_full']

# Codificadores
le_sint = LabelEncoder()
y_sint_full = le_sint.fit_transform(df_sint['label_full'])

le_common = LabelEncoder()
le_common.fit(clases_comunes)
y_sint_common = le_common.transform(df_sint_common['label_common'])
y_nhanes_common = le_common.transform(df_nhanes_common['label_common'])

# Definir X_sint_common (las mismas columnas comunes)
X_sint_common = df_sint_common[cols_comunes].copy()
X_nhanes_common = df_nhanes_common[cols_comunes].copy()

# =============================================================================
# 5. Dividir datasets (70% entrenamiento, 20% prueba, 10% validación)
# =============================================================================
# Sintético completo (para escenario 1)
X_sint_train, X_sint_temp, y_sint_train, y_sint_temp = train_test_split(
    X_sint, y_sint_full, test_size=0.3, stratify=y_sint_full, random_state=42)
X_sint_val, X_sint_test, y_sint_val, y_sint_test = train_test_split(
    X_sint_temp, y_sint_temp, test_size=2/3, stratify=y_sint_temp, random_state=42)

# Sintético común (para entrenar en escenario 2)
X_sint_common_train, X_sint_common_temp, y_sint_common_train, y_sint_common_temp = train_test_split(
    X_sint_common, y_sint_common, test_size=0.3, stratify=y_sint_common, random_state=42)
X_sint_common_val, X_sint_common_test, y_sint_common_val, y_sint_common_test = train_test_split(
    X_sint_common_temp, y_sint_common_temp, test_size=2/3, stratify=y_sint_common_temp, random_state=42)

# NHANES común (solo test para escenario 2)
X_nh_train, X_nh_test, y_nh_train, y_nh_test = train_test_split(
    X_nhanes_common, y_nhanes_common, test_size=0.2, stratify=y_nhanes_common, random_state=42)

print(f"\n--- Tamaños de conjuntos ---")
print(f"Sintético (full): train={X_sint_train.shape[0]}, val={X_sint_val.shape[0]}, test={X_sint_test.shape[0]}")
print(f"Sintético común: train={X_sint_common_train.shape[0]}, val={X_sint_common_val.shape[0]}, test={X_sint_common_test.shape[0]}")
print(f"NHANES común: train={X_nh_train.shape[0]}, test={X_nh_test.shape[0]}")

# =============================================================================
# 6. Preprocesamiento: imputación y escalado (aunque los datos ya están en [0,1])
# =============================================================================
def preprocess(X_train, X_val, X_test):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imputer.fit_transform(X_train))
    X_va = scaler.transform(imputer.transform(X_val)) if X_val is not None else None
    X_te = scaler.transform(imputer.transform(X_test))
    return X_tr, X_va, X_te

# =============================================================================
# 7. Modelos con hiperparámetros
# =============================================================================
def get_model(name, class_weight=None):
    if name == 'Decision Tree':
        return DecisionTreeClassifier(max_depth=8, random_state=42, class_weight=class_weight)
    elif name == 'SVM':
        return SVC(C=1000, gamma=0.01, random_state=42, probability=True, class_weight=class_weight)
    elif name == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.1)

# =============================================================================
# 8. Función de entrenamiento y evaluación
# =============================================================================
def train_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, scenario_name):
    X_tr, X_va, X_te = preprocess(X_train, X_val, X_test)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"\n--- {scenario_name} | {model.__class__.__name__} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Macro : {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
    return {'model': model.__class__.__name__, 'scenario': scenario_name,
            'acc': acc, 'f1': f1_macro}

# =============================================================================
# 9. Escenario 1: Sintético → Sintético (con diabetes)
# =============================================================================
results = []
for name in ['Decision Tree', 'SVM', 'MLP']:
    model = get_model(name, class_weight='balanced' if name != 'MLP' else None)
    res = train_evaluate(model, X_sint_train, y_sint_train, X_sint_val, y_sint_val,
                         X_sint_test, y_sint_test, le_sint, "Sintético → Sintético")
    results.append(res)

# =============================================================================
# 10. Escenario 2: Sintético → NHANES (clases comunes)
# =============================================================================
for name in ['Decision Tree', 'SVM', 'MLP']:
    model = get_model(name, class_weight='balanced' if name != 'MLP' else None)
    res = train_evaluate(model, X_sint_common_train, y_sint_common_train, X_sint_common_val, y_sint_common_val,
                         X_nh_test, y_nh_test, le_common, "Sintético → NHANES")
    results.append(res)

# =============================================================================
# 11. Tabla resumen y visualización
# =============================================================================
df_res = pd.DataFrame([{'Modelo': r['model'], 'Escenario': r['scenario'],
                        'Accuracy': r['acc'], 'F1-Macro': r['f1']} for r in results])
print("\n" + "="*70)
print("RESUMEN DE RESULTADOS")
print("="*70)
print(df_res.to_string(index=False))

# Gráfico comparativo
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_res, x='Modelo', y='Accuracy', hue='Escenario', palette='Set2')
ax.set_ylim(0, 1.05)
ax.set_title('Comparación de Accuracy por Modelo y Escenario (datos normalizados)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()

# =============================================================================
# 12. Interpretación del domain shift
# =============================================================================
print("\n" + "="*70)
print("INTERPRETACIÓN DEL DOMAIN SHIFT")
print("="*70)
for model_name in df_res['Modelo'].unique():
    acc_syn = df_res[(df_res['Modelo']==model_name) & (df_res['Escenario']=='Sintético → Sintético')]['Accuracy'].values[0]
    acc_real = df_res[(df_res['Modelo']==model_name) & (df_res['Escenario']=='Sintético → NHANES')]['Accuracy'].values[0]
    deg = (acc_syn - acc_real) * 100
    print(f"{model_name}: Degradación Accuracy = {deg:.2f} puntos porcentuales")
    if deg > 20:
        print("  → Domain shift severo. Los datos sintéticos no generalizan a NHANES.")
    elif deg > 10:
        print("  → Domain shift moderado. Se recomienda entrenar con datos reales.")
    else:
        print("  → Domain shift leve. Los datos sintéticos tienen buena capacidad de generalización.")