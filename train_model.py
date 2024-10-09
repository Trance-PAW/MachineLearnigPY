import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Cargar los datos del archivo CSV
df = pd.read_csv('solana_historical_data.csv')

# Crear tres EMAs de 15, 20 y 50 periodos
df['EMA_15'] = df['close'].ewm(span=15, adjust=False).mean()
df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

# Calcular el RSI (Relative Strength Index)
window_length = 14
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Calcular el MACD (Moving Average Convergence Divergence)
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Calcular el SAR (Parabolic Stop and Reverse)
df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

# Crear una columna 'target' para predecir si el precio subirá en las próximas 5 horas
df['target'] = (df['close'].shift(-5) > df['close']).astype(int)

# Seleccionar las características para el modelo
X = df[['EMA_15', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'SAR']]
y = df['target']

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Definir el pipeline con un imputer, escalador y SVM
pipeline_svm = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Definir los hiperparámetros para GridSearch
param_grid_svm = {
    'classifier__C': [16, 18, 20],
    'classifier__kernel': ['rbf'],
    'classifier__gamma': [0.9, 0.1, 0.5, 0.7, 'scale'],
    'classifier__class_weight': ['balanced']  # Dar peso a la clase minoritaria
}

# Realizar GridSearchCV para encontrar los mejores parámetros
grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_svm.fit(X_train, y_train)

# Imprimir los mejores parámetros y el mejor score
print(f"Mejor score SVM: {grid_search_svm.best_score_}")
print(f"Mejores parámetros SVM: {grid_search_svm.best_params_}")

# Evaluar el modelo en el conjunto de prueba
test_score = grid_search_svm.score(X_test, y_test)
print(f"Score en prueba: {test_score}")

# Guardar el pipeline con los mejores parámetros
best_pipeline_file = 'best_pipeline_svm.sav'
joblib.dump(grid_search_svm.best_estimator_, best_pipeline_file)

print(f"Mejor pipeline guardado en {best_pipeline_file}")

