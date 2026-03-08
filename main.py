import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# =========================================
# 1. Obtención de los datos
# =========================================

path = kagglehub.dataset_download("henrysue/online-shoppers-intention")

print(f"La carpeta está en: {path}")

path = kagglehub.dataset_download("henrysue/online-shoppers-intention")
file_path = os.path.join(path, "online_shoppers_intention.csv")

# =========================================
# 2. Configuración de los datos
# =========================================

df = pd.read_csv(
    file_path,
    sep=None,
    engine='python',
    encoding='latin1',
    on_bad_lines='skip'
)

print(f"Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")

# =========================================
# 3. Procesado de los datos
# =========================================

df = df.drop_duplicates()

y = df['Revenue']
y = y.astype(int)

X_encoded = pd.get_dummies(df.drop('Revenue', axis=1), columns=['Month', 'VisitorType'], drop_first=True)
X_encoded['Weekend'] = X_encoded['Weekend'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# =========================================
# 4. Entrenamiento del modelo y resultados
# =========================================

rf = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=300, max_depth=20, min_samples_split=10, max_features='sqrt')
rf.fit(X_train, y_train)
probs = rf.predict_proba(X_test)[:, 1]

threshold = 0.5
pred_umbral = (probs >= threshold).astype(int)

print("Resultados:")
print(classification_report(y_test, pred_umbral))
