# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:51:38 2020

@author: luhoe
"""
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
import time

class TrainingPlot(Callback):
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
        self.delta_time = []
        fig , (ax1, ax2, ax3) = plt.subplots(3) 
        fig.tight_layout() 
        self.figNumber = plt.gcf().number
        self.starttime = time.time()

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.delta_time.append(time.time() - self.starttime)
        # Before plotting ensure at least 2 epochs have passed

        if len(self.losses) > 0:
            N = np.arange(0, len(self.losses))
            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            #plt.figure()
            #fig, (ax1, ax2) = plt.subplots(2)

            fig = plt.figure(self.figNumber)
            ax = fig.get_axes()
            ax1 = ax[0]
            ax2 = ax[1]
            ax3 = ax[2]
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax1.plot(N, self.losses, label = "train_loss")
            ax2.plot(N, self.acc, label = "train_acc")
            ax1.plot(N, self.val_losses, label = "val_loss")
            ax2.plot(N, self.val_acc, label = "val_acc")
            ax3.plot(N, self.delta_time, label = "Timing")
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


# # For visualization
class Images(Callback):
    
    def __init__(self, validation_data):
        super(Callback, self).__init__()
        self.validation_data = validation_data
        self.indizes = np.random.randint(self.validation_data[0].shape[0], size=8)
        fig1 , ax = plt.subplots(2,8) 
        self.figNumber = plt.gcf().number
    
    def on_epoch_end(self, epoch, logs):
          indices = self.indizes
          test_data = self.validation_data[0][indices]
          pred_data = self.model.predict(test_data)
          
          fig1 = plt.figure(self.figNumber)
          ax = fig1.get_axes()
          
          #initial data
          ax1 = ax[0]
          ax2 = ax[1]
          ax3 = ax[2]
          ax4 = ax[3]
          ax5 = ax[4]
          ax6 = ax[5]
          ax7 = ax[6]
          ax8 = ax[7]          
          ax1.imshow(test_data[0,:,:])
          ax2.imshow(test_data[1,:,:])
          ax3.imshow(test_data[2,:,:])
          ax4.imshow(test_data[3,:,:])
          ax5.imshow(test_data[4,:,:])
          ax6.imshow(test_data[5,:,:])
          ax7.imshow(test_data[6,:,:])
          ax8.imshow(test_data[7,:,:])

          #predicted data          
          ax9 = ax[8]
          ax10 = ax[9]
          ax11 = ax[10]
          ax12 = ax[11]
          ax13 = ax[12]
          ax14 = ax[13]
          ax15 = ax[14]
          ax16 = ax[15]          
          ax9.imshow(pred_data[0,:,:])
          ax10.imshow(pred_data[1,:,:])
          ax11.imshow(pred_data[2,:,:])
          ax12.imshow(pred_data[3,:,:])
          ax13.imshow(pred_data[4,:,:])
          ax14.imshow(pred_data[5,:,:])
          ax15.imshow(pred_data[6,:,:])
          ax16.imshow(pred_data[7,:,:])
          
          plt.pause(0.05)
