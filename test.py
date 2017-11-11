import keras.utils
from keras.applications import VGG16
# from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import time

MAX = 20
bar = keras.utils.Progbar(MAX)
for i in range(MAX):
    time.sleep(1)
    bar.update(i)
