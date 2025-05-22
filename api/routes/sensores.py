from fastapi import APIRouter, Request
from core.firebase_config import db


router = APIRouter()

@router.post("/sensores")
async def guardar_datos_sensores(request: Request):
    # Obtener los datos del cuerpo de la solicitud
    datos = await request.json()

    # Guardar los datos en Firestore
    try:
        # Aquí puedes agregar la lógica para guardar los datos en Firestore
        # Por ejemplo, usando el cliente de Firestore que has inicializado
        db.collection("lecturas_sensores").add(datos)
        return {"mensaje": "Datos guardados correctamente"}
    except Exception as e:
        return {"error": str(e)}