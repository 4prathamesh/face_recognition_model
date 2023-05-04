

import torch
import torch.nn as nn
import numpy as np
from keras.models import load_model

# Load the trained Keras model
keras_model = load_model('best_model.h5')

# Define the equivalent PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 32)
        self.layer5 = nn.Linear(32, 16)
        self.layer6 = nn.Linear(16, 8)
        self.layer7 = nn.Linear(8, 4)
        self.layer8 = nn.Linear(4, 2)
        self.layer9 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = nn.functional.relu(x)
        x = self.layer4(x)
        x = nn.functional.relu(x)
        x = self.layer5(x)
        x = nn.functional.relu(x)
        x = self.layer6(x)
        x = nn.functional.relu(x)
        x = self.layer7(x)
        x = nn.functional.relu(x)
        x = self.layer8(x)
        x = nn.functional.relu(x)
        x = self.layer9(x)
        x = self.activation(x)
        return x

# Copy over the weights from the Keras model to the PyTorch model
pytorch_model = PyTorchModel()
pytorch_model.layer1.weight.data = torch.from_numpy(keras_model.layers[0].get_weights()[0].T)
pytorch_model.layer1.bias.data = torch.from_numpy(keras_model.layers[0].get_weights()[1])
pytorch_model.layer2.weight.data = torch.from_numpy(keras_model.layers[1].get_weights()[0].T)
pytorch_model.layer2.bias.data = torch.from_numpy(keras_model.layers[1].get_weights()[1])
pytorch_model.layer3.weight.data = torch.from_numpy(keras_model.layers[2].get_weights()[0].T)
pytorch_model.layer3.bias.data = torch.from_numpy(keras_model.layers[2].get_weights()[1])
pytorch_model.layer4.weight.data = torch.from_numpy(keras_model.layers[3].get_weights()[0].T)
pytorch_model.layer4.bias.data = torch.from_numpy(keras_model.layers[3].get_weights()[1])
pytorch_model.layer5.weight.data = torch.from_numpy(keras_model.layers[4].get_weights()[0].T)
pytorch_model.layer5.bias.data = torch.from_numpy(keras_model.layers[4].get_weights()[1])
pytorch_model.layer6.weight.data = torch.from_numpy(keras_model.layers[5].get_weights()[0].T)
pytorch_model.layer6.bias.data = torch.from_numpy(keras_model.layers[5].get_weights()[1])
pytorch_model.layer7.weight.data = torch.from_numpy(keras_model.layers[6].get_weights()[0].T)
pytorch_model.layer7.bias.data = torch.from_numpy(keras_model.layers[6].get_weights()[1])
pytorch_model.layer8.weight.data = torch.from_numpy(keras_model.layers[7].get_weights()[0].T)
pytorch_model.layer8.bias.data = torch.from_numpy(keras_model.layers[7].get_weights()[1])
pytorch_model.layer9.weight.data = torch.from_numpy(keras_model.layers[8].get_weights()[0].T)
pytorch_model.layer9.bias.data = torch.from_numpy(keras_model.layers[8].get_weights()[1])

# Convert PyTorch model to Torch format and save in .t7 file
torch.save(pytorch_model.state_dict(), 'trained_weights.t7', _use_new_zipfile_serialization=False)
torch.save(pytorch_model, 'face_recognition_model.t7')


'''# Save only the state dictionary
torch.save(pytorch_model.state_dict(), 'trained_weights.pth')
pytorch_model.load_state_dict(torch.load('trained_weights.pth'))

torch.save(pytorch_model, 'face_recognition_model.pt')'''
