from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app=FastAPI()

#endpoint='http://localhost:8501/v1/models/potatoes_model:predict'


MODEL=tf.keras.models.load_model('models_potato/1')
CLASS_NAMES=['EARLY BLIGHT','LATE BLIGHT','HEALTHY']


@app.get('/ping')
async def ping():
    return 'Hello!! server is alive!!'


def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    img_batch=np.expand_dims(image,0)# since the 'model.predict' doesnt accept single image, but btaches of images => expand dimension of that image
    
    
    prediction=MODEL.predict(img_batch)
    index=np.argmax(prediction[0])
    predicted_class=CLASS_NAMES[index]
    confidence=np.max(prediction[0])
    return {
        'class':predicted_class,
        'confidence':float(confidence)
    }


@app.post('/predict')
async def predict(
    file: UploadFile
):
    #bytes=await file.read()
    image=read_file_as_image(await file.read())
    return image



# 'async' and 'await' helps in managinf the process where large number of users send large images and cause delays in processing

if __name__=='__main__':
    uvicorn.run(app, host='localhost',port= 8000)

# tf_serving helps in dynamically manage traffic between different versions of the model..
# this is especially useful, if one wants to switch/divert, say 10% traffic from a production to a 'beta' model etc..