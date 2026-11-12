from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'housing.csv')
X, y = df.drop(columns='sale_price'), df['sale_price']
num = X.select_dtypes(include='number').columns.tolist(); cat = X.select_dtypes(exclude='number').columns.tolist()
prep = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num), ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
models = {'baseline_mean': DummyRegressor(strategy='mean'), 'linear_regression': LinearRegression(), 'random_forest': RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1)}
results, best_name, best_rmse, best_pipe = {}, None, float('inf'), None
for name, model in models.items():
 pipe = Pipeline([('preprocess', prep), ('model', model)]); pipe.fit(X_train, y_train); pred = pipe.predict(X_test)
 rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
 results[name] = {'mae': float(mean_absolute_error(y_test, pred)), 'rmse': rmse, 'r2': float(r2_score(y_test, pred))}
 if rmse < best_rmse: best_name, best_rmse, best_pipe = name, rmse, pipe
out = ROOT / 'artifacts'; out.mkdir(exist_ok=True); joblib.dump(best_pipe, out / 'house_price_model.joblib')
(out / 'metrics.json').write_text(json.dumps({'best_model': best_name, 'models': results}, indent=2))
print(json.dumps({'best_model': best_name, 'metrics': results}, indent=2))
