import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split

def load_dataset() -> pd.DataFrame:
    df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, "henrysue/online-shoppers-intention", "online_shoppers_intention.csv")
    print('✅ Conjunto de datos cargado correctamente!')
    return df

def preprocess_base(data: pd.DataFrame, month_map: dict) -> pd.DataFrame:
    df = data.copy()

    if 'Month' in df.columns:
        df['Month'] = df['Month'].map(month_map)
    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)
    if 'Revenue' in df.columns:
        df['Revenue'] = df['Revenue'].astype(int)
        
    return df

def generate_user_from_json(user_info: dict):
    id = user_info['id']
    user_dict = user_info['user_dict'].copy()
    user_dict['Weekend'] = user_info['user_dict'] == 'True'
    return id, user_dict


def prepare_train_data(df_raw: pd.DataFrame, month_map: dict):
    df = df_raw.drop_duplicates()
    
    df = preprocess_base(df, month_map)
    
    y = df['Revenue']
    X_raw = df.drop('Revenue', axis=1)
    
    X = pd.get_dummies(X_raw, columns=['VisitorType', 'OperatingSystems', 'Browser', 'Region', 'TrafficType'], 
                       drop_first=True, dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_columns = X_train.columns.tolist()

    print('✅ Conjunto de datos de entrenamiento preparado correctamente!')
    
    return X_train, X_test, y_train, y_test, train_columns

def show_dataset_summary(df: pd.DataFrame):
    print(f' - Conjunto de datos formado por {df.shape[0]} filas y {df.shape[1]} columnas')

