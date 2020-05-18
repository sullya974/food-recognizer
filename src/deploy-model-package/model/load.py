# The loader function
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from tensorflow.keras.models import model_from_json
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from tensorflow.keras.models import load_model

tf.keras.backend.clear_session()

def init():
  sess = tf.Session()
  graph = tf.get_default_graph()

  MNIST_model_json_file_path = os.path.join("model", "MNIST_model.json")
  MNIST_model_h5_file_path = os.path.join("model", "MNIST_model.h5")
  global loaded_model_json
  with open(MNIST_model_json_file_path, "r") as json_file:
      loaded_model_json = json_file.read()

  # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
  # Otherwise, their weights will be unavailable in the threads after the session there has been set
  set_session(sess)
  
  # use Keras model_from_json to make a loaded model
  global loaded_model
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  
  loaded_model.load_weights(MNIST_model_h5_file_path)
  print("Loaded Model from disk")
  
  # compile and evaluate loaded model
  loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

  # global graph
  # graph = tf.compat.v1.get_default_graph()
  # graph = tf.get_default_graph()

  return loaded_model, graph

def init_model_loading():
  # Load the best save model to make predictions
  tf.keras.backend.clear_session()
  sess = tf.Session()
  graph = tf.get_default_graph()

  # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
  # Otherwise, their weights will be unavailable in the threads after the session there has been set
  set_session(sess)

  MNIST_model_h5_file_path = os.path.join("model", "MNIST_model_trained.hdf5")
  model = load_model(MNIST_model_h5_file_path, compile=False)  
  return model, graph  


# def load_predictor():
#   global predictor
  
#   MNIST_model_json_file_path = os.path.join("model", "MNIST_model.json")
#   MNIST_model_h5_file_path = os.path.join("model", "MNIST_model.h5")
  
#   predictor = ktrain.load_predictor(MNIST_model_h5_file_path)
  
#   if hasattr(predictor.model, '_make_predict_function'):
#     predictor.model._make_predict_function()
    
#   global graph
#   graph = tf.compat.v1.get_default_graph()

#   return predictor, graph