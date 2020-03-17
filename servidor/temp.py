# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import librosa
import librosa.display
import numpy as np
from sklearn.preprocessing import LabelEncoder
import keras
import librosa
from flask import Flask, jsonify, request
import json
import wget
from os import remove
from flask import Flask
from flask_cors import CORS



NuevoModelo=keras.models.load_model('C:/Users/Felipe-Veloza/PruebasLicor/modeloAudios.h5')

le = LabelEncoder()
max_pad_len = 174

def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


"""# Set the path to the full UrbanSound dataset 
fulldatasetpath = 'C:/Users/Felipe-Veloza/PruebasLicor/Audios'

metadata = pd.read_csv('C:/Users/Felipe-Veloza/PruebasLicor/DataAudios3.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.join(os.path.abspath(fulldatasetpath),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class_name"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

y = np.array(featuresdf.class_label.tolist())
yy = to_categorical(le.fit_transform(y)) 
np.save('C:/Users/Felipe-Veloza/PruebasLicor/le.npy',le.classes_)
"""
le.classes_=np.load('C:/Users/Felipe-Veloza/PruebasLicor/le.npy')

Miclasse=[]
respuesta=[]
def MiPrediccion(file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, 40, 174, 1)

    predicted_vector = NuevoModelo.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    respuesta.append(predicted_class[0])
    
    predicted_proba_vector = NuevoModelo.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        Miclasse.append(((category[0],format(predicted_proba[i], '.32f'))))




def DescargarAudio(url):
    try:
        wget.download(url,'./AudiosParaProcesar/Miarchivo.wav')
    except:
        print("Fallo")        

app = Flask(__name__)

CORS(app)

@app.route("/")
def helloWorld():
  return "Hello, cross-origin-world!"

@app.route('/api/audios', methods=['POST'])
def add_task():
    url=request.get_json()['url']
    DescargarAudio(url)
    filename = 'C:/Users/Felipe-Veloza/PruebasLicor/AudiosParaProcesar/Miarchivo.wav'
    MiPrediccion(filename)
    remove(filename)
    return jsonify({"result": respuesta.pop(0)})

@app.route('/', methods=['GET'])
def prueba():
        return jsonify({"result": "Hola mundo"})
    
# @app.route('/api/report', methods=['POST'])
# def reporte():
#     if(respuesta=="Lider-agua" or respuesta=="Nectar-agua" or respuesta=="Antio-agua")
#     return jsonify({"result": "Aguardiente Adulterado"})


if __name__ == '__main__':
    app.run(host='localhost', threaded=False)

