import firebase_admin
from firebase_admin import credentials, firestore
import os

# Ruta al archivo json de credenciales
ruta_credenciales = os.path.join(os.path.dirname(__file__), '../../serviceAccountKey.json')

# Inicializar la aplicación de Firebase (sólo si no está inicializada)

if not firebase_admin._apps:
    cred = credentials.Certificate(ruta_credenciales)
    firebase_admin.initialize_app(cred)

    #Cliente de Firestore
    db = firestore.client()