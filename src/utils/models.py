from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from typing import Union
import pandas as pd
from utils.data import preprocess_base


ValidModel = Union[RandomForestClassifier, XGBClassifier]

def evaluate_model(model: ValidModel, X, y):
    preds = model.predict(X)
    print(classification_report(y, preds))
    return preds


def train_random_forest(X, y, best_config: bool = True, custom_config: dict = None) -> RandomForestClassifier:
    if best_config:
        rf = RandomForestClassifier(
            random_state=42, 
            n_estimators=300, 
            max_depth=20, 
            min_samples_split=10, 
            max_features='sqrt', 
            class_weight='balanced'
        )
    else:
        config = custom_config if custom_config else {}
        rf = RandomForestClassifier(**config)
    
    rf.fit(X, y)

    return rf


def train_xgboost(X, y, best_config: bool = True, custom_config: dict = None) -> XGBClassifier:
    if best_config:
        xgb = XGBClassifier(
            colsample_bytree=1.0, 
            learning_rate= 0.1, 
            max_depth= 3, 
            n_estimators= 200, 
            subsample= 1.0
        )
    else:
        config = custom_config if custom_config else {}
        xgb = XGBClassifier(**config)
    
    xgb.fit(X, y)

    return xgb

def predict_new_user(user_dict: dict, model: ValidModel, train_columns: list[str], month_map):
    df_user = pd.DataFrame([user_dict])
    df_user = preprocess_base(df_user, month_map)
    df_user_final = df_user.reindex(columns=train_columns, fill_value=0)

    prediction = model.predict(df_user_final)[0]
    probability = model.predict_proba(df_user_final)[0][1]

    if prediction == 1:
        print(f'Este usuario es POTENCIAL COMPRADOR con una seguridad del {probability*100:.2f}%')
    else:
        print(f'Este usuario [NO] es POTENCIAL COMPRADOR con una seguridad del {(1-probability)*100:.2f}%')
    
    return prediction, probability



