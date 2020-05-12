# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:51:38 2020

@author: luhoe
"""
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

class TrainingPlot(Callback):
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        fig , (ax1, ax2) = plt.subplots(2) 
        fig.tight_layout() 

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        # Before plotting ensure at least 2 epochs have passed

        if len(self.losses) > 0:
            N = np.arange(0, len(self.losses))
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            #plt.figure()
            #fig, (ax1, ax2) = plt.subplots(2)

            fig = plt.gcf()
            ax = fig.get_axes()
            ax1 = ax[0]
            ax2 = ax[1]
            ax1.cla()
            ax2.cla()
            ax1.plot(N, self.losses, label = "train_loss")
            ax2.plot(N, self.acc, label = "train_acc")
            ax1.plot(N, self.val_losses, label = "val_loss")
            ax2.plot(N, self.val_acc, label = "val_acc")
            ax1.set(xlabel="Epoch #", ylabel="Loss")
            ax2.set(xlabel="Epoch #", ylabel="Accuracy")
            ax1.set_title("Training Loss and Validation Loss")
            ax2.set_title("Training accuracy and Validation accuracy")
            ax1.legend(loc="upper right")
            ax2.legend(loc="lower right")
            #plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            #plt.savefig('output/Epoch-{}.png'.format(epoch))
            plt.pause(0.05)
            #plt.show(block=False)
            #plt.close()
