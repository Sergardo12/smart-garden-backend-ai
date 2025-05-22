# smart-garden-backend-ai
Este repositorio será para desarrolLar el backend junto con el procesamiento inteligente de nuestra aplicaciones Smart Garden


## Arquitectura del proyecto:

smart_garden_ia_backend/
│
├── api/                      # Endpoints de la API (usaremos FastAPI o Flask)
│   ├── routes/               # Módulos separados por funcionalidad
│   │   ├── sensores.py       # Recibe datos de sensores (POST)
│   │   ├── predicciones.py   # Devuelve predicción de riego o riesgo
│   │   └── control.py        # Comandos para actuar (riego ON/OFF)
│   └── main.py               # Inicia el servidor API
│
├── core/                     # Núcleo de lógica del sistema
│   ├── models/               # Modelos de ML entrenados (pickle, .h5, etc.)
│   │   └── predictor_riego.pkl
│   ├── logic/                # Código para cargar y usar modelos
│   │   ├── predictor.py      # Usa el modelo para hacer predicciones
│   │   ├── riesgos.py        # Lógica para detectar anomalías
│   │   └── controlador.py    # Decisión de riego automático/manual
│   └── vision/               # Procesamiento de imágenes (crecimiento planta)
│       └── analizador.py
│
├── data/                     # Datos recolectados y usados para entrenar
│   ├── sensores.csv
│   ├── imagenes/             # Imágenes para análisis de crecimiento
│   └── reportes_generados/
│
├── training/                 # Scripts para entrenar modelos
│   ├── preparar_datos.py
│   ├── entrenar_riego.py
│   └── entrenar_vision.py
|
|____firebase/
|    |___serviceAccountKey.json
│
├── chatbot/                  # Lógica para chatbot de sugerencias
│   └── chat.py               # (opcionalmente usando NLP simple o reglas)
│
├── utils/                    # Funciones auxiliares comunes
│   └── helpers.py
│
├── requirements.txt          # Librerías necesarias (FastAPI, scikit-learn, etc.)
└── README.md




### 🧠 ¿Cómo funciona tu sistema desde el sensor hasta la IA?
🟢 1. Captura de datos (ESP32)
Tus sensores (temperatura, humedad, luz, etc.) están conectados a un ESP32.

El ESP32 envía datos directamente a Firestore, por ejemplo a una colección "sensores_crudos".

Esos datos llegan sin procesar.

➡️ No pasan por tu backend todavía.

🔵 2. Lectura y procesamiento en el backend
Tu backend (smart_garden_ia_backend/) entra en acción aquí.

En especial el archivo:

bash
Copy
Edit
api/routes/sensores.py
Desde ahí puedes:

a) Leer los datos crudos de Firebase:
python
Copy
Edit
datos = db.collection("sensores_crudos").stream()
b) Procesarlos:
Lógica en core/logic/predictor.py o riesgos.py.

Ejemplo: predecir si se debe regar o si hay riesgo de plaga.

c) Guardar los resultados:
python
Copy
Edit
db.collection("predicciones").add(resultado)
🔴 3. ¿Cómo entrenas tu modelo?
Para eso tienes la carpeta:

bash
Copy
Edit
training/
│   ├── preparar_datos.py        # Limpia, filtra y convierte datos de sensores en CSV
│   ├── entrenar_riego.py        # Entrena el modelo para predecir riego
│   └── entrenar_vision.py       # Entrena la IA de visión (plantas)
📂 Fuente de datos de entrenamiento:
Puede venir de:

La colección "sensores_crudos" de Firestore

Archivos CSV guardados en data/sensores.csv

Imágenes en data/imagenes/

🧠 Resultado del entrenamiento:
Guardas tu modelo en:

bash
Copy
Edit
core/models/predictor_riego.pkl
Luego lo usas en:
python
Copy
Edit
from core.logic.predictor import predecir_riego
🟣 4. ¿Cómo entra la visión computacional?
Si tienes fotos del crecimiento de las plantas (tomadas por una cámara y subidas al ESP32 o una app móvil), las guardas en:

bash
Copy
Edit
data/imagenes/
Tu módulo core/vision/analizador.py analiza esas imágenes.

Y tu script training/entrenar_vision.py entrena un modelo (por ejemplo con OpenCV, TensorFlow, etc.).

🟡 5. ¿Y el chatbot?
El chatbot está en chatbot/chat.py.

Podría recibir preguntas del usuario por API (api/routes/chat.py, que puedes crear).

Este chatbot puede:

Responder consejos (“¿Qué hacer si mi planta tiene hojas amarillas?”)

Recomendar riego o fertilizante según predicciones previas.

Leer datos desde Firebase o el backend.

🗂️ ¿Tu estructura de carpetas es adecuada?
Sí, está muy bien pensada. Cada cosa tiene su lugar:

Carpeta	Propósito
api/routes/	Entradas a tu sistema, vía HTTP (lo que el usuario o ESP32 usa)
core/logic/	Procesos de decisión y análisis
core/models/	Modelos entrenados (.pkl, .h5)
core/vision/	Análisis de imágenes
training/	Scripts para entrenar modelos
data/	Archivos de entrada/salida, imágenes, CSV
chatbot/	Lógica de chatbot
firebase/	Configuración Firebase (clave privada)

🧭 Flujo resumido de tu proyecto
text
Copy
Edit
[ESP32 + sensores] → [Firestore (datos crudos)] → [Backend lee y predice] 
→ [Firestore (datos procesados)] → [Chatbot o Dashboard lee resultados]
Y aparte:

text
Copy
Edit
[Datos históricos (Firestore o CSV)] → [Entrenamiento de modelo ML] 
→ [Modelo .pkl] → [Backend usa modelo para nuevas predicciones]



### Librerias importantes en el proyecto proyecto dsmart garden 

✅ 1. NumPy
📌 ¿Para qué sirve?

Manipulación de arrays y operaciones numéricas.

🧠 Dónde lo usarás:

Cuando prepares datos de sensores para entrenar un modelo.

En la visión computacional (procesar imágenes como arrays de pixeles).

python
Copy
Edit
import numpy as np

valores = np.array([23.5, 45.1, 33.8])
promedio = np.mean(valores)
✅ 2. Pandas
📌 ¿Para qué sirve?

Leer, limpiar y manipular datos tabulares (como CSV).

🧠 Dónde lo usarás:

En training/preparar_datos.py para convertir los datos de Firestore a CSV o DataFrames.

Para analizar tendencias de riego, humedad, etc.

python
Copy
Edit
import pandas as pd

df = pd.read_csv("data/sensores.csv")
df_limpio = df.dropna()
✅ 3. Scikit-learn (sklearn)
📌 ¿Para qué sirve?

Entrenamiento de modelos de ML clásicos (Regresión, Árboles, SVM...).

🧠 Dónde lo usarás:

En training/entrenar_riego.py para crear el modelo .pkl.

En core/logic/predictor.py para cargarlo y hacer predicciones.

python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
✅ 4. TensorFlow y Keras
📌 ¿Para qué sirve?

Deep Learning, redes neuronales, visión computacional.

🧠 Dónde lo usarás:

En training/entrenar_vision.py para entrenar una red neuronal que detecte crecimiento o problemas en plantas a partir de fotos.

Modelo se guarda como .h5 o .pb.

python
Copy
Edit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    ...
])
✅ 5. OpenCV (cv2)
📌 ¿Para qué sirve?

Procesar imágenes, detectar contornos, colores, cambios visuales.

🧠 Dónde lo usarás:

En core/vision/analizador.py para analizar imágenes en tiempo real o comparar crecimiento entre días.

python
Copy
Edit
import cv2

imagen = cv2.imread("data/imagenes/planta1.jpg")
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
✅ ¿Debes instalar todas estas librerías?
Sí, y debes agregarlas en tu requirements.txt para que cualquier colaborador pueda instalar tu entorno fácilmente.

Ejemplo mínimo de requirements.txt:
nginx
Copy
Edit
fastapi
uvicorn
firebase-admin
pandas
numpy
scikit-learn
opencv-python
tensorflow
🧩 Conclusión: ¿Las necesitas?
Sí, todas estas librerías son esenciales para:

Manipular datos crudos (pandas, numpy)

Entrenar y usar modelos (sklearn, tensorflow)

Procesar imágenes (opencv)

Conectar todo con la IA y tu backend (fastapi, firebase-admin)


