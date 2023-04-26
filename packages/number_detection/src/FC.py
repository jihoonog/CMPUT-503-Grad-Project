import numpy as np

class NP_model():
    # Only supports MLP linear models
    def __init__(self, model_weight):
        self.weights = model_weight

    def show_dims(self):
        for key, value in self.weights.items():
            print(key, value.shape)

    def predict(self, x):
        temp = x
        for key, value in self.weights.items():
            if "weight" in key:
                temp = np.matmul(temp, value.transpose())
            elif "bias" in key:
                temp = temp + value
                temp = np.maximum(temp, 0)
        return temp