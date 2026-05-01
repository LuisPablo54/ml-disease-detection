
"""
Entrenamiento de modelos con datos reales (NHANES) y balanceo de clases
- Modelos: DecisionTree (max_depth=8), SVM (C=1000, γ=0.01), MLP (128-64-32)
- División: 70% train, 20% test, 10% validation
- Técnicas de balanceo: Sin balanceo, SMOTE, ADASYN
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

from imblearn.over_sampling import SMOTE, ADASYN

np.random.seed(42)

#  
# 1. Cargar y normalizar NHANES
#  
df_nhanes = pd.read_csv("data_test/NHANES_2017_2020_labeled_diseases.csv")
print(f"NHANES shape original: {df_nhanes.shape}")

# Rangos fisiológicos para normalizar (solo columnas que existen)
ranges = {
    "Platelets": (150000, 450000),
    "White Blood Cells": (4000, 11000),
    "Red Blood Cells": (4.2, 5.4),
    "Hematocrit": (38, 52),
    "Mean Corpuscular Volume": (80, 100),
    "Mean Corpuscular Hemoglobin": (27, 33),
    "Mean Corpuscular Hemoglobin Concentration": (32, 36),
    "HDL Cholesterol": (40, 60),
    "ALT": (10, 40),
    "Heart Rate": (60, 100)
}

df_norm = df_nhanes.copy()
for col, (min_val, max_val) in ranges.items():
    if col in df_norm.columns:
        df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        df_norm[col] = df_norm[col].clip(0, 1)

#  
# 2. Seleccionar columnas comunes (las mismas 10 usadas en el análisis anterior)
#  
cols_comunes = ['ALT', 'HDL Cholesterol', 'Heart Rate', 'Hematocrit',
                'Mean Corpuscular Hemoglobin', 'Mean Corpuscular Hemoglobin Concentration',
                'Mean Corpuscular Volume', 'Platelets', 'Red Blood Cells', 'White Blood Cells']

# Verificar que todas existan
for c in cols_comunes:
    if c not in df_norm.columns:
        raise ValueError(f"Columna {c} no encontrada en NHANES")

X = df_norm[cols_comunes].copy()
# Mapeo de etiquetas (solo clases comunes, excluyendo 'borderline')
mapa_label = {
    'anemia': 'anemia',
    'healthy': 'healthy',
    'borderline': 'other',
    'thalassemia': 'thalassemia',
    'thrombocytopenia': 'thrombocytopenia'
}
df_norm['label'] = df_norm['condition'].map(mapa_label)
df_clean = df_norm[df_norm['label'] != 'other'].copy()
X = df_clean[cols_comunes]
y_str = df_clean['label']

# Codificar etiquetas
le = LabelEncoder()
y = le.fit_transform(y_str)
print(f"\nClases y frecuencias originales:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {(y == i).sum()}")

#  
# 3. División train/validation/test (70/10/20) estratificada
# =========================================================================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)  # 0.125 de 0.8 = 0.1 total

print(f"\nTamaños: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")

#  
# 4. Preprocesamiento: imputar y escalar
#  
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

def preprocess(X_train, X_val, X_test, fit_on_train=True):
    if fit_on_train:
        X_tr = scaler.fit_transform(imputer.fit_transform(X_train))
        X_va = scaler.transform(imputer.transform(X_val))
        X_te = scaler.transform(imputer.transform(X_test))
    else:
        # Para datos ya preprocesados externamente (SMOTE/ADASYN)
        X_tr = scaler.fit_transform(imputer.fit_transform(X_train))
        X_va = scaler.transform(imputer.transform(X_val))
        X_te = scaler.transform(imputer.transform(X_test))
    return X_tr, X_va, X_te

# Aplicar preprocesamiento base (sin balanceo aún)
X_train_proc, X_val_proc, X_test_proc = preprocess(X_train, X_val, X_test)

#  
# 5. Definición de modelos con hiperparámetros óptimos
#  
def get_model(name, class_weight=None):
    if name == 'Decision Tree':
        return DecisionTreeClassifier(max_depth=8, random_state=42, class_weight=class_weight)
    elif name == 'SVM':
        return SVC(C=1000, gamma=0.01, random_state=42, probability=True, class_weight=class_weight)
    elif name == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42,
                             early_stopping=True, validation_fraction=0.1)

#  
# 6. Función de entrenamiento y evaluación
#  
def train_evaluate(model, X_tr, y_tr, X_va, y_va, X_te, y_te, label_encoder, scenario_name):
    # Para MLP, si tiene early_stopping, no usamos X_va explícitamente, pero lo pasamos igual
    if isinstance(model, MLPClassifier):
        # MLP usa validation_fraction internamente, no acepta X_val en fit
        model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
    print(f"\n--- {scenario_name} | {model.__class__.__name__} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Macro : {f1_macro:.4f}")
    if hasattr(model, 'classes_'):
        print(classification_report(y_te, y_pred, target_names=label_encoder.classes_, zero_division=0))
    else:
        print(classification_report(y_te, y_pred, zero_division=0))
    return acc, f1_macro

#  
# 7. Escenarios de balanceo: Sin balanceo, SMOTE, ADASYN
#  
balance_methods = {
    'Sin balanceo': None,
    'SMOTE': 'smote',
    'ADASYN': 'adasyn'
}

results = []

for method_name, method_type in balance_methods.items():
    print("\n" + "="*70)
    print(f"Balanceo: {method_name}")
    print("="*70)
    
    if method_type is None:
        X_bal = X_train_proc
        y_bal = y_train
    elif method_type == 'smote':
        sm = SMOTE(random_state=42)
        X_bal, y_bal = sm.fit_resample(X_train_proc, y_train)
        print(f"SMOTE: tamaño original={len(y_train)} → nuevo={len(y_bal)}")
    elif method_type == 'adasyn':
        ada = ADASYN(random_state=42)
        X_bal, y_bal = ada.fit_resample(X_train_proc, y_train)
        print(f"ADASYN: tamaño original={len(y_train)} → nuevo={len(y_bal)}")
    
    for model_name in ['Decision Tree', 'SVM', 'MLP']:
        # Para modelos que soportan class_weight, lo usamos solo en 'Sin balanceo' para ayudar.
        # Con SMOTE/ADASYN, el balanceo ya está en los datos, por lo que class_weight=None.
        if method_name == 'Sin balanceo' and model_name != 'MLP':
            model = get_model(model_name, class_weight='balanced')
        else:
            model = get_model(model_name, class_weight=None)
        
        acc, f1 = train_evaluate(model, X_bal, y_bal, X_val_proc, y_val, X_test_proc, y_test, le,
                                 f"{method_name} - {model_name}")
        results.append({
            'Balanceo': method_name,
            'Modelo': model_name,
            'Accuracy': acc,
            'F1-Macro': f1
        })


# 8. Tabla resumen y visualización
# -*- coding: utf-8 -*-
"""
Código mejorado para visualización de resultados:
- Gráfico de barras agrupadas (Accuracy y F1-Macro)
- Heatmap de resultados
- Radar chart por modelo
- Gráfico de degradación relativa
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import pi

# =============================================================================
# Datos de ejemplo (reemplazar con resultados reales)
# =============================================================================
# Estructura esperada del DataFrame con resultados
df_res = pd.DataFrame([
    # Balanceo, Modelo, Accuracy, F1-Macro
    ['Sin balanceo', 'Decision Tree', 0.85, 0.72],
    ['Sin balanceo', 'SVM', 0.88, 0.75],
    ['Sin balanceo', 'MLP', 0.86, 0.73],
    ['SMOTE', 'Decision Tree', 0.87, 0.78],
    ['SMOTE', 'SVM', 0.89, 0.80],
    ['SMOTE', 'MLP', 0.88, 0.79],
    ['ADASYN', 'Decision Tree', 0.86, 0.76],
    ['ADASYN', 'SVM', 0.88, 0.79],
    ['ADASYN', 'MLP', 0.87, 0.77]
], columns=['Balanceo', 'Modelo', 'Accuracy', 'F1-Macro'])

print("Datos cargados:")
print(df_res)

# =============================================================================
# Configuración general de estilo
# =============================================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
colors = {'Sin balanceo': '#E74C3C', 'SMOTE': '#2ECC71', 'ADASYN': '#3498DB'}
markers = {'Sin balanceo': 'o', 'SMOTE': 's', 'ADASYN': '^'}

# =============================================================================
# Gráfico 1: Barras agrupadas (Accuracy y F1-Macro)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Comparación de Modelos por Técnica de Balanceo', fontsize=16, fontweight='bold')

# Accuracy
sns.barplot(data=df_res, x='Modelo', y='Accuracy', hue='Balanceo', 
            palette='Set2', ax=axes[0], edgecolor='black', linewidth=0.8)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_xlabel('Modelo', fontsize=12)
axes[0].set_title('Accuracy por Modelo y Balanceo', fontsize=14, fontweight='bold')
axes[0].legend(title='Balanceo', loc='lower right', fontsize=10)
# Añadir valores sobre las barras
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.3f', fontsize=9, padding=3)

# F1-Macro
sns.barplot(data=df_res, x='Modelo', y='F1-Macro', hue='Balanceo', 
            palette='Set2', ax=axes[1], edgecolor='black', linewidth=0.8)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel('F1-Macro', fontsize=12)
axes[1].set_xlabel('Modelo', fontsize=12)
axes[1].set_title('F1-Macro por Modelo y Balanceo', fontsize=14, fontweight='bold')
axes[1].legend(title='Balanceo', loc='lower right', fontsize=10)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt='%.3f', fontsize=9, padding=3)

plt.tight_layout()
plt.savefig('Fase_3/img/comparacion_modelos_balanceo.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Gráfico 2: Heatmap de resultados (Accuracy y F1)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Heatmap de Resultados por Modelo y Balanceo', fontsize=16, fontweight='bold')

# Heatmap Accuracy
pivot_acc = df_res.pivot(index='Modelo', columns='Balanceo', values='Accuracy')
sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', 
            linewidths=0.5, ax=axes[0], cbar_kws={'label': 'Accuracy'})
axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Técnica de Balanceo', fontsize=12)
axes[0].set_ylabel('Modelo', fontsize=12)

# Heatmap F1-Macro
pivot_f1 = df_res.pivot(index='Modelo', columns='Balanceo', values='F1-Macro')
sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='YlOrRd', 
            linewidths=0.5, ax=axes[1], cbar_kws={'label': 'F1-Macro'})
axes[1].set_title('F1-Macro', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Técnica de Balanceo', fontsize=12)
axes[1].set_ylabel('Modelo', fontsize=12)

plt.tight_layout()
plt.savefig('Fase_3/img/heatmap_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Interpretación de resultados

print("Resultado")
for model in df_res['Modelo'].unique():
    print(f"\n{model}:")
    for method in df_res['Balanceo'].unique():
        row = df_res[(df_res['Modelo']==model) & (df_res['Balanceo']==method)]
        acc = row['Accuracy'].values[0]
        f1 = row['F1-Macro'].values[0]
        print(f"  {method}: Accuracy={acc:.4f}, F1={f1:.4f}")