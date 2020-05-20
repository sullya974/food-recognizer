# from tensorflow.python.keras.models import load_model
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

def init(path_trained_model):
  K.clear_session()
  model = load_model(path_trained_model, compile=False)

  return model

def init_food_classes():
  return ['green_beans', 'chicken_wings', 'zucchini', 'peanuts', 'beef_carpaccio', 'broccoli', 'brussels_sprouts', 
  'peking_duck', 'caprese_salad', 'omelette', 'strawberries', 'high_fat_sauce', 'kale', 'baby_back_ribs', 'pork_chop', 
  'mussels', 'walnuts', 'cherries', 'scallops', 'oysters', 'plum', 'fish', 'beef_tartare', 'pecan_nuts', 'octopus', 
  'seafood', 'squid', 'butter', 'caesar_salad', 'seaweed_salad', 'blackberries', 'clementine', 'cauliflower', 'cabbage', 
  'avocado', 'pulled_pork', 'raspberries', 'prime_rib', 'greek_salad', 'blueberries', 'beet_salad', 'peach', 'cantaloupe', 
  'oil', 'eggs', 'asparagus', 'clams', 'cheese', 'kiwi', 'filet_mignon', 'steak', 'almonds', 'brazil_nuts', 'guacamole', 
  'chicken_curry', 'spinach', 'macadamia', 'hazel_nuts']
  