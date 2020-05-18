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

tensorflow.keras.backend.clear_session()

# global graph
# global model
# global model_best, graph_best

# model, graph = init()
# model._make_predict_function()

# K.clear_session()
# model_best, graph_best = init_model_loading()

app = Flask(__name__)
# run_with_ngrok(app)

@app.route("/")
def index_view():
  return render_template("index.html")
  # return "<h1>Running Flask on Google Colab</h1>"

def convertImage(imgData1):
    # print(imgData1)
    imgstr = re.search('base64,(.*)', imgData1).group(1)
    with open("output.png", "wb") as output:
        output.write(base64.b64decode(imgstr))

def predict_class(model, images, show=True):
    pass
#   for img in images:
#     # print(img)
#     img = image.load_img(img, target_size=(299, 299))
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img /= 255.

#     pred = model.predict(img)
#     # index = np.argmax(pred)
#     # food_list = os.listdir("src/g_img_dataset/test")
#     # food_list.sort()
#     # pred_value = food_list[index]
#     print(pred)
#     # if show:
#     #   plt.imshow(img[0])
#     #   plt.axis("off")
#     #   plt.title(pred_value)
#     #   plt.show()        

# @app.route("/predict/", methods=["GET", "POST"])
@app.route("/predict/")
@app.route('/uploader/', methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join("static", "images_validation", secure_filename(f.filename)))
        print('file uploaded successfully')
        
        # imgData = request.get_data()
        # print(imgData)
        # convertImage(imgData)
        food_img = secure_filename(f.filename)
        food_img_path = os.path.join("static", "images_validation", food_img)
        
        K.clear_session()
        
        img = image.load_img(food_img_path, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.
        
        model = load_model("model/best_model_58class_all.hdf5", compile=False)
        
        pred = model.predict(img)
        index = np.argmax(pred)
        food_classes = os.listdir("/home/sully/code/practice/portfolio_data/food-recognizer/datasets/google_images_dataset/images/test")
        food_classes.sort()
        pred_value = food_classes[index]
        
        print(pred_value)
        
        return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_value.capitalize().replace("_", " "))
        
    print(">> random")
    # food_img_path = ""
    # pred_value = ""
    # return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_value.capitalize().replace("_", " "))

    # convertImage(imgData)
    food_validation_list = os.listdir("static/images_validation/")
    food_img = food_validation_list[np.random.randint(len(food_validation_list))]
    food_img_path = os.path.join("static", "images_validation", food_img)

    K.clear_session()
    img = image.load_img(food_img_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    model = load_model("model/best_model_58class_all.hdf5", compile=False)
    
    pred = model.predict(img)
    index = np.argmax(pred)
    food_classes = os.listdir("/home/sully/code/practice/portfolio_data/food-recognizer/datasets/google_images_dataset/images/test")
    food_classes.sort()
    pred_value = food_classes[index]

    # "pred" retourne une liste  10 élément de probabilité de prédiction de valeurs
    print(pred_value)
    return render_template("draw.html", food_image = os.path.join("..", food_img_path), food_pred = pred_value.capitalize().replace("_", " "))
    


    # K.clear_session()
    # # model_best = load_model("model/MNIST_model_trained.hdf5", compile=False)
    # model_best = load_model("model/best_model_58class_all.hdf5", compile=False)
    # pred = model_best.predict(x)
    # # "pred" retourne une liste  10 élément de probabilité de prédiction de valeurs
    # print(pred)

    # #     # # Récupère l'index de l'élément ayant la plus grande probabilité prédite
    # index = np.argmax(pred)

    # food_list = os.listdir("/home/sully/code/practice/portfolio_data/food-recognizer/datasets/google_images_dataset/images/test")

    # food = food_list[index] # Récupère la valeur "digit" correspondant à l'index

    # print(food)

    # return food

    
# def predict():
#     print(">> imgData")
    
#     imgData = request.form["my_hidden"]
#     print(imgData)
#     convertImage(imgData)

#     x = cv2.imread('output.png', cv2.IMREAD_GRAYSCALE)
#     dim = (28, 28)
#     # x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
#     x = cv2.resize(x, dim, interpolation=cv2.INTER_AREA)
#     print("Resized shape : ", x.shape)
#     x = x.reshape(1,28,28,1)


#     K.clear_session()
#     # model_best = load_model("model/MNIST_model_trained.hdf5", compile=False)
#     model_best = load_model("model/MNIST_best_model.hdf5", compile=False)
#     pred = model_best.predict(x)
#     # "pred" retourne une liste  10 élément de probabilité de prédiction de valeurs
#     print(pred)

#     # digits_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#     # # Récupère l'index de l'élément ayant la plus grande probabilité prédite
#     # index = np.argmax(pred)
#     # digit = digits_list[index] # Récupère la valeur "digit" correspondant à l'index

#     # print("Valeur prédite : ", str(digit))

    
#     # predict_class(model_best, images, True)
#     # session = K.get_session()
#     # graph = tensorflow.compat.v1.get_default_graph()
#     # graph.finalize()
    
#     # with graph_best.as_default():
#     #     pred = model_best.predict(x)
#     #     print(pred)

#     # with graph.as_default():
#     #     out = model.predict(x)
#     #     print(out)
#     #     print(np.argmax(out, axis=1))

#     # resized = cv2.resize(x, dim, interpolation=cv2.INTER_AREA)
#     # cv2.imshow("Resized", resized)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


#     # x = imageio.imread('output.png')
#     # x = np.invert(x)
    
#     # im = Image.fromarray(x)
#     # size = tuple((np.array(im.size) * 784).astype(int))
#     # x = np.array(im.resize(size))
#     # x = x.reshape(1,28,28,1)

  

# #   with graph.asdefault():
# #     out = model.predict(x)
# #     print(out)
# #     print(np.argmax(out, axis=1))

# #     response = np.array_str(np.argmax(out,axis=1))
# #     return response
    
#     return "None"
    

# #   print(imgData)


# #   #-----------


if __name__ == "__main__":
    app.run()