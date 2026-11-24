import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# extraccion de datos desde el archivo csv y limpieza de valores na
datos_entrenamiento = pd.read_csv("sensores_planta.csv")
datos_entrenamiento = datos_entrenamiento.dropna()

# Columnas del df usadas para entrenamiento
columnas_entrenamiento = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
X = datos_entrenamiento[columnas_entrenamiento]

# Columna del df para predecir resultado
y = datos_entrenamiento["estado"]


entrenamiento_X, validacion_X, entrenamiento_y, validacion_y = train_test_split(X, y, train_size=0.8, test_size=0.2) # Division del dataset 80-20 entrenamiento-validacion

modelo = RandomForestClassifier(random_state=1)
modelo.fit(entrenamiento_X, entrenamiento_y) # Entrenamiento del modelo

# Prediccion y metrica de precision
predicc_y = modelo.predict(validacion_X)
precision = accuracy_score(validacion_y, predicc_y)

print(f"Accuracy: {precision:.2f}")
if precision >= 0.80:
    joblib.dump(modelo, "modelo_mantenimiento.pkl")
    print("Precision superior al 80%; guardando modelo")
else:
    print("La precision no alcanzo el requisito, intente de nuevo")

