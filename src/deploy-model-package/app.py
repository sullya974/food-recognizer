from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("./model"))
from load import *
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from werkzeug import secure_filename
from werkzeug.routing import Rule

K.clear_session()

global model, food_classes

app = Flask(__name__)

path_trained_model = "model/best_model_58class_all.hdf5"
model = init(path_trained_model)
food_classes = init_food_classes()

# Gère le routing
app.url_map.add(Rule('/', endpoint="predict"))
app.url_map.add(Rule('/predict/', endpoint="predict"))

# IMPORTANT : Décommenter la ligne ci-dessous pour exécuter NGROK et sonner accès à l'application en ligne
# run_with_ngrok(app)

def do_prediction(food_img_path):
    K.clear_session()
    img = image.load_img(food_img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    
    pred = model.predict(img)
    
    indexes = np.argsort(pred[0])[::-1][:3]

    food_probas = np.array(pred[0])[indexes]
    food_probas = [round(proba*100, 2) if idx == 0 else round(proba*100, 5) for idx, proba in enumerate(food_probas)]
    
    food_classes.sort()
    
    pred_values = [food_classes[index].capitalize().replace("_", " ") for index in indexes]
    
    print(pred_values)
    
    return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_values, food_probas = food_probas)

# @app.route("/")
# @app.route("/predict/")
# @app.route('/uploader/', methods=["GET", "POST"])
@app.endpoint('predict')
def predict():
    path_images_validation = os.path.join("static", "images_validation")
    path_list_food_test = os.path.join("/", "home" , "sully" , "code", "practice", "portfolio_data", "food-recognizer", "datasets", "google_images_dataset", "images", "test")
    
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(path_images_validation, secure_filename(f.filename)))
        print('file uploaded successfully')
        
        food_img = secure_filename(f.filename)
        food_img_path = os.path.join(path_images_validation, food_img)
    else:
        food_validation_list = os.listdir(path_images_validation)
        food_img = food_validation_list[np.random.randint(len(food_validation_list))]
        food_img_path = os.path.join(path_images_validation, food_img)

    
    return do_prediction(food_img_path)

if __name__ == "__main__":
    app.run()