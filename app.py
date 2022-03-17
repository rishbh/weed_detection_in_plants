from time import sleep
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2


st.set_page_config(page_title="Weed Seedings Detection")
st.title("Detecting Weed Seedlings in Crops")

info_md = Path("app.md").read_text()
st.markdown(info_md)

classes = {0: 'Black-grass', 1: 'Charlock', 2: 'Cleavers', 3: 'Common Chickweed',  5: 'Fat Hen',
           6: 'Loose Silky-bent',  8: 'Scentless Mayweed', 9: 'Shepherd\'s Purse', 10: 'Small-flowered Cranesbill',
           11: 'Sugar beet', 4: 'Common wheat', 7: 'Maize'}

weeds = {'Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Fat Hen',
         'Loose Silky-bent',  'Scentless Mayweed', 'Shepherd\'s Purse', 'Small-flowered Cranesbill',
         'Sugar beet'}

crops = {'Common wheat', 'Maize'}

TFLITE_MODEL = Path("model.tflite")

with st.spinner("Please wait initialising the model"):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL.__str__())
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]


uploader = st.file_uploader("", type=['jpg', 'png'])


def predict(image):
    with st.spinner("Computing ..."):
        interpreter.set_tensor(input_details["index"], image)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])
    prediction = np.argmax(out, axis=1)[0]
    return classes[prediction]


if uploader is not None:
    file = uploader.read()
    img = Image.open(uploader)
    img = mobilenet_v2.preprocess_input(np.array(img.resize((224, 224))))
    img = np.expand_dims(img, axis=0)
    p = predict(img)
    st.image(
        file, caption=f"{p} - {'Weed' if p in weeds else 'Not a weed'}", width=500)
