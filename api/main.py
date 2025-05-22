from fastapi import FastAPI
from api.routes import sensores

app = FastAPI()

#Registrar las rutas
app.include_router(sensores.router)