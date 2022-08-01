import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import os

use_cuda = t.cuda.is_available()
csv_file = os.path.dirname(os.path.realpath(__file__)) + '/data.csv'
batch_size = 32


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data, test_data = ChallengeDataset(train_data, 'train'), ChallengeDataset(test_data,'test')
train_data, test_data = t.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True), \
                        t.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)
# create an instance of our ResNet model
model = model.ResNet()
#

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCELoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-6)
trainer = Trainer(model, criterion, optimizer,
                  train_data, test_data, use_cuda, 300)

# go, go, go... call fit on trainer
res = trainer.fit(2)
onnx_file_path = "/home/cip/medtech2020/dy28wofo/ExPytorch/models/model.onnx"
# save the model
with open(onnx_file_path, "wb+") as path:
    trainer.save_onnx(path)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')