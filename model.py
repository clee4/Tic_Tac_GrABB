import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import model_from_yaml, model_from_json, load_model
from PIL import Image, ImageOps

class XOModel:
    def __init__(self, JSON_path, h5_path):
        # load YAML and create model
        json_file = open(JSON_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights(h5_path)
        print("Loaded model from disk")

        self.classes = {0:'O',1:'X'}

    def identify(self, img, show=False):
        """
        Identifies if an image is of an x or an o

        img: a 28x28 pixel grayscale image containing an x or an o
        show: boolean value that outputs confidence level
        
        returns string of x or an o
        """
        img = img_to_array(img)
        img = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        pred = self.model.predict(img)

        if show:
            print("Prediction: "+self.classes[pred.argmax()])
            print("Confidence: "+str(pred[0][pred.argmax()]*100)+"%")

        return self.classes[pred.argmax()].lower()

class TicTacModel:
    def __init__(self, h5_path):
        # loads model from storage
        self.model = load_model(h5_path)
    
    def pick_move(self, board):
        """
        Picks best move based on loaded ML Model
        
        board: list containing each space's occupancy

        returns index of best move based on model
        """
        pre = self.model.predict(np.asarray([self.one_hot(board)]), batch_size=1)[0]
        print(pre)
        highest = -1000
        num = -1
        for j in range(9):
            if board[j][-1] == ' ' and pre[j] > highest:
                highest = pre[j].copy()
                num = j
        return num

    def one_hot(self, state):
        """
        Remaps tic tac toe grid state to a 27 index long list
        Every three spaces correspond to one position in the grid

        state: list containing the current state of the tic tac toe grid

        returns remapped list
        """
        current_state = []

        for position in state:
            if position[-1] == ' ':
                current_state.append(1)
                current_state.append(0)
                current_state.append(0)
            elif position[-1] == 'x':
                current_state.append(0)
                current_state.append(1)
                current_state.append(0)
            elif position[-1] == 'o':
                current_state.append(0)
                current_state.append(0) 
                current_state.append(1)
        
        return current_state

if __name__ == "__main__":
    import cv2
    model = XOModel(os.path.join('storage', 'model.yaml'), os.path.join('storage', 'model.h5'))
    img = Image.open('7.png').convert("L")
    
    prediction = model.identify(img)
    print(type(prediction), prediction)