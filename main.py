import torch
import torch.nn as nn
import numpy as np
from keras.models import load_model
from keras.layers import Dense


# Load the trained Keras model
keras_model = load_model('models/best_model.h5')

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self, num_layers):
        super(PyTorchModel, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(128, 128))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(128, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Copy over the weights from the Keras model to the PyTorch model
pytorch_model = PyTorchModel(num_layers=len(keras_model.layers))
for i, keras_layer in enumerate(keras_model.layers):
    if isinstance(keras_layer, Dense):
        pytorch_layer = pytorch_model.layers[i * 2]
        pytorch_layer.weight.data = torch.from_numpy(keras_layer.get_weights()[0].T)
        pytorch_layer.bias.data = torch.from_numpy(keras_layer.get_weights()[1])

# Convert PyTorch model to Torch format and save in .t7 file
torch.save(pytorch_model.state_dict(), 'trained_weights.t7', _use_new_zipfile_serialization=False)
torch.save(pytorch_model, 'face_recognition_model.t7')



