from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
#from keras.preprocessing import image
import keras.utils as image
import numpy as np

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title='CatDog Recognition')
st.title('Cat and Dog Classification')
st.markdown(""" 
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> 
    """, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_best_model():
    model = keras.models.load_model('animal_model.h5')

    model.make_predict_function()          # Necessary
    print('Model loaded. Start serving...')
    return model

st.markdown('You can find the Convolutional Neural Netowrk used [here](https://github.com/nnhwin)')


st.subheader('Classify the image')
image_file = st.file_uploader('Choose the Image', ['jpg', 'png'])

if image_file is not None:
    x = image.img_to_array(image_file)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    model=get_best_model()
    classes = model.predict(images, batch_size=10)
    print(classes)
    if classes[0]>0:
        prediction = 'Dog'
    else:
        prediction = 'Cat'
    st.write(f'The image is predicted as {prediction}')

st.subheader('Classify the image as Cat or Dog')
sentence_image_files = st.file_uploader('Select the Images', ['jpg', 'png'], accept_multiple_files = True)
