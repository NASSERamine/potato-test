from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import uvicorn

app = FastAPI()

Model = tf.keras.models.load_model("mon_modele4.keras")
class_name = ["Potato___Early_blight", "Potato___healthy", "Potato___Late_blight"]

def read_fil_as_image(data) -> np.ndarray:
    # Ouvrir l'image à partir des données binaires
    image = Image.open(BytesIO(data))
    
    # Convertir l'image en tableau numpy
    image = image.convert("RGB")  # Assure-toi que l'image est en mode RGB
    image_array = np.array(image)
    
    # Normaliser les valeurs des pixels si nécessaire (ajuster si nécessaire selon le modèle)
    image_array = image_array / 255.0
    
    return image_array

@app.get("/ping")
async def ping():
    return "hello i am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_fil_as_image(await file.read())
    
    # Vérifier que l'image n'est pas None
    if image is None:
        return {"error": "L'image n'a pas pu être lue."}

    image_batch = np.expand_dims(image, axis=0)
    
    predictions = Model.predict(image_batch)
    print(predictions)
    index = np.argmax(predictions[0])
    predicted = class_name[index]
    
    return {"prediction": predicted}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
