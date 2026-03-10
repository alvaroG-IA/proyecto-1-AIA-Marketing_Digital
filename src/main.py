from utils.data import load_dataset, prepare_train_data, show_dataset_summary
from utils.models import train_random_forest, train_xgboost, evaluate_model, predict_new_user
from utils.interfaz import seleccionar_modelo

# =========================================
# 1. Carga de conjunto de datos
# =========================================

df = load_dataset()

# =========================================
# 3. Procesado de los datos
# =========================================
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
X_train, X_test, y_train, y_test, train_columns = prepare_train_data(df, month_map)
show_dataset_summary(X_train)

# =========================================
# 4. Entrenamiento del modelo y resultados
# =========================================

opt = seleccionar_modelo()
if opt == 1:
    model = train_random_forest(X=X_train, y=y_train, best_config=True)
elif opt == 2:
    model = train_xgboost(X=X_train, y=y_train, best_config=True)

evaluate_model(model, X_test, y_test)

nuevo_ejemplo = {
    'Administrative': 2, 'Administrative_Duration': 45.0,
    'Informational': 0, 'Informational_Duration': 0.0,
    'ProductRelated': 15, 'ProductRelated_Duration': 650.5,
    'BounceRates': 0.00, 'ExitRates': 0.02,
    'PageValues': 15.4, 'SpecialDay': 0.0,
    'Month': 'Nov', 'OperatingSystems': 2,
    'Browser': 2, 'Region': 1, 'TrafficType': 2,
    'VisitorType': 'Returning_Visitor', 'Weekend': False
}

pred, prob = predict_new_user(nuevo_ejemplo, model, train_columns, month_map)