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

df_clean = df.drop_duplicates()

meses_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
df_clean['Month'] = df_clean['Month'].map(meses_map)
df_clean = pd.get_dummies(df_clean, columns=['VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'], drop_first=True, dtype=int)
df_clean['Weekend'] = df_clean['Weekend'].astype(int)
df_clean['Revenue'] = df_clean['Revenue'].astype(int)

y = df_clean['Revenue']
X = df_clean.drop('Revenue', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =========================================
# 4. Entrenamiento del modelo y resultados
# =========================================

rf = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=20, min_samples_split=10, max_features='sqrt', class_weight='balanced')
rf.fit(X_train, y_train)
probs = rf.predict_proba(X_test)[:, 1]

threshold = 0.5
pred_umbral = (probs >= threshold).astype(int)

print("Resultados:")
print(classification_report(y_test, pred_umbral))
