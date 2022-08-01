import torch as t
from sklearn.metrics import f1_score
import numpy as np

class Trainer:
    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, path):
        torch_model = self._model.cpu()
        torch_model.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(torch_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss

    def val_test_step(self, x, y):
        # predict
        output = self._model(x)
        # propagate through the network and calculate the loss and predictions
        loss = self._crit(output, y)
        # return the loss and the predictions
        return loss, output

    def train_epoch(self):
        # set training mode
        self._model.train(True)
        losses = []
        # iterate through the training set
        for sample in self._train_dl:
            x, y = sample[0], sample[1]
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            loss = self.train_step(x, y)
            losses.append(loss.item())

        # calculate the average loss for the epoch and return it
        total_loss = sum(losses) / len(losses)
        return total_loss

    def val_test(self):
        # set eval mode
        self._model.eval()
        losses = []
        f_ones = []
        # disable gradient computation
        with t.no_grad():
            # iterate through the validation set
            for sample in self._val_test_dl:
                x, y = sample[0], sample[1]
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, prediction = self.val_test_step(x, y)
                losses.append(loss.item())

                if self._cuda:
                    prediction = prediction.cpu().numpy()
                    y = y.cpu().numpy()
                f_ones.append(f1_score(y, np.where(prediction > 0.5, 1, 0), average="macro"))

                # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_loss = sum(losses) / len(losses)
        avg_f_one = sum(f_ones) / len(f_ones)
        # return the loss and print the calculated metrics
        print("Average loss during validation: ", avg_loss)
        print("Average f11 during validation: ", avg_f_one)
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        epoch_counter = 0
        early_stopping_counter = 0
        train_losses = []
        validation_losses = []

        # stop by epoch number
        while epochs == -1 or epoch_counter < epochs:
            # train for a epoch and then calculate the loss and metrics on the validation set
            epoch_counter += 1
            print("Epoch: ", epoch_counter)
            training_loss = self.train_epoch()
            validation_loss = self.val_test()

            # append the losses to the respective lists
            train_losses.append(training_loss)
            validation_losses.append(validation_loss)

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if validation_loss < validation_losses[-1]:
                early_stopping_counter = 0
                self.save_checkpoint(epoch_counter)
            else:
                early_stopping_counter += 1

            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if early_stopping_counter >= self._early_stopping_patience:
                break

        # return the losses for both training and validation
        return train_losses, validation_losses
