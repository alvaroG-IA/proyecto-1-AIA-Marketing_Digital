import kagglehub
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report
import shutil
import os
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Obtener la ruta donde kagglehub guarda los datos
path = kagglehub.dataset_download("henrysue/online-shoppers-intention")

print(f"La carpeta está en: {path}")

# Eliminar la carpeta completa para resetear el archivo dañado (utilizarlo si da error)
'''
try:
    shutil.rmtree(path)
    print("Carpeta eliminada con éxito. Ya se puede volver a descargar el dataset.")
except Exception as e:
    print(f"No se pudo eliminar: {e}")
'''

path = kagglehub.dataset_download("henrysue/online-shoppers-intention")
file_path = os.path.join(path, "online_shoppers_intention.csv")

# 2. Leer con configuración "todoterreno"
df = pd.read_csv(
    file_path,
    sep=None,             # Detecta automáticamente si es coma o punto y coma
    engine='python',      # El motor de Python es más flexible con errores de formato
    encoding='latin1',    # Evita el error de Unicode original
    on_bad_lines='skip'   # Si una línea está rota, la salta en lugar de detenerse
)

# 3. Verificar
print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")

df = df.drop_duplicates()

y = df['Revenue']
y = y.astype(int)

X_encoded = pd.get_dummies(df.drop('Revenue', axis=1), columns=['Month', 'VisitorType'], drop_first=True)
X_encoded['Weekend'] = X_encoded['Weekend'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Distribución original: {Counter(y_train)}")
print(f"Nueva distribución: {Counter(y_train_res)}")

print('1) Usando Random Forest:')
rf = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=300, max_depth=20, min_samples_split=5, max_features='sqrt')
rf.fit(X_train, y_train)
probs = rf.predict_proba(X_test)[:, 1]

nuevo_umbral = 0.5
pred_umbral = (probs >= nuevo_umbral).astype(int)

print(classification_report(y_test, pred_umbral))

print('2) Usando XGBoost:')
xgbmodel = xgb.XGBClassifier(random_state=42)
xgbmodel.fit(X_train_res, y_train_res)
probs = xgbmodel.predict_proba(X_test)[:, 1]

nuevo_umbral = 0.5
pred_umbral_2 = (probs >= nuevo_umbral).astype(int)

print(classification_report(y_test, pred_umbral_2))

# Tratándolo como que la compra sea un Outlier
print('3) Usando Isolation Forest')
iso_forest = IsolationForest(contamination=0.15, random_state=42)
iso_forest.fit(X_train, y_train)
pred_3 = iso_forest.predict(X_test)
pred_3_final = [1 if x == -1 else 0 for x in pred_3]
print(classification_report(y_test, pred_3_final))

importances_rf = rf.feature_importances_
importances_xgb = xgbmodel.feature_importances_
indices = X_encoded.columns

# 2. Crear un DataFrame para facilitar la visualización
feature_imp_df = pd.DataFrame({
    'Feature': indices,
    'Random Forest': importances_rf,
    'XGBoost': importances_xgb
}).sort_values(by='Random Forest', ascending=False)

# 3. Graficar (Top 10)
plt.figure(figsize=(15, 5))
top_10_df = feature_imp_df.head(10).melt(id_vars='Feature', var_name='Modelo', value_name='Importancia')

sns.barplot(data=top_10_df, y='Importancia', x='Feature', hue='Modelo', palette='magma')

plt.title('¿En qué columnas se fijan más los modelos? (Top 10)', fontsize=15)
plt.xlabel('Nivel de Importancia')
plt.ylabel('Variables del Dataset')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

errores = X_test.copy()
errores['Real'] = y_test
errores['Pred_RF'] = pred_umbral

# Filtramos solo donde el modelo se equivocó (Falsos Negativos: Iban a comprar y el modelo dijo que no)
falsos_negativos = errores[(errores['Real'] == 1) & (errores['Pred_RF'] == 0)]
falsos_positivos = errores[(errores['Real'] == 0) & (errores['Pred_RF'] == 1)]

print(f"El modelo no detectó {len(falsos_negativos)} compras.")
print(f"Falsos positivos {len(falsos_positivos)}.")
print("Promedio de PageValues en los errores:", falsos_negativos['PageValues'].mean())