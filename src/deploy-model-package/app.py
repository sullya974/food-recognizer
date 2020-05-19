from PIL import Image
import imageio
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
# from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os
import base64
sys.path.append(os.path.abspath("./model"))
from load import *
import cv2
import tensorflow
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug import secure_filename
from werkzeug.routing import Rule

tensorflow.keras.backend.clear_session()

app = Flask(__name__)
app.url_map.add(Rule('/', endpoint="predict"))
app.url_map.add(Rule('/predict/', endpoint="predict"))
# run_with_ngrok(app)

# @app.route("/")
# def index_view():
#   return render_template("draw.html")
#   # return "<h1>Running Flask on Google Colab</h1>"

def convertImage(imgData1):
    # print(imgData1)
    imgstr = re.search('base64,(.*)', imgData1).group(1)
    with open("output.png", "wb") as output:
        output.write(base64.b64decode(imgstr))

def do_prediction(food_img_path, path_trained_model, path_list_food_test):
    K.clear_session()
    img = image.load_img(food_img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    
    model = load_model(path_trained_model, compile=False)
    
    pred = model.predict(img)
    index = np.argmax(pred)
    food_classes = os.listdir(path_list_food_test)
    food_classes.sort()
    pred_value = food_classes[index]
    
    print(pred_value)
    
    return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_value.capitalize().replace("_", " "))        

# @app.route("/")
# @app.route("/predict/")
# @app.route('/uploader/', methods=["GET", "POST"])
@app.endpoint('predict')
def predict():
    path_images_validation = os.path.join("static", "images_validation")
    path_list_food_test = os.path.join("/", "home" , "sully" , "code", "practice", "portfolio_data", "food-recognizer", "datasets", "google_images_dataset", "images", "test")
    path_trained_model = "model/best_model_58class_all.hdf5"
    
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(path_images_validation, secure_filename(f.filename)))
        print('file uploaded successfully')
        
        food_img = secure_filename(f.filename)
        food_img_path = os.path.join(path_images_validation, food_img)
        
        return do_prediction(food_img_path, path_trained_model, path_list_food_test)
        # K.clear_session()
        # img = image.load_img(food_img_path, target_size=(299, 299))
        # img = image.img_to_array(img)
        # img = np.expand_dims(img, axis=0)
        # img /= 255.
        
        # model = load_model(path_trained_model, compile=False)
        
        # pred = model.predict(img)
        # index = np.argmax(pred)
        # food_classes = os.listdir(path_list_food_test)
        # food_classes.sort()
        # pred_value = food_classes[index]
        
        # print(pred_value)
        
        # return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_value.capitalize().replace("_", " "))
        
    print(">> random")
    food_validation_list = os.listdir(path_images_validation)
    food_img = food_validation_list[np.random.randint(len(food_validation_list))]
    food_img_path = os.path.join(path_images_validation, food_img)

    return do_prediction(food_img_path, path_trained_model, path_list_food_test)
    # K.clear_session()
    # img = image.load_img(food_img_path, target_size=(299, 299))
    # img = image.img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # img /= 255.

    # model = load_model(path_trained_model, compile=False)
    
    # pred = model.predict(img)
    # index = np.argmax(pred)
    # food_classes = os.listdir(path_list_food_test)
    # food_classes.sort()
    # pred_value = food_classes[index]

    # # "pred" retourne une liste  10 élément de probabilité de prédiction de valeurs
    # print(pred_value)
    # return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_value.capitalize().replace("_", " "))


if __name__ == "__main__":
    app.run()