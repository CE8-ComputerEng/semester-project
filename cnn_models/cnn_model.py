
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import includes.utils as utils
from includes.utils import plot_confusion_matrix, plot_multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

class CNNClassifier(pl.LightningModule):
    def __init__(self, classes, sample_shape, spectrogram_type):
        super(CNNClassifier, self).__init__()

        self.sample_shape = list(sample_shape)
        self.classes = classes
        num_classes = len(classes)
        self.spectrogram_type = spectrogram_type
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        
        self.sample_shape[0] = (self.sample_shape[0]//2//2//2//2//2)
        self.sample_shape[1] = (self.sample_shape[1]//2//2//2//2//2)

        self.fc1 = nn.Linear(256 * self.sample_shape[0] * self.sample_shape[1], 512)

        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.Sigmoid()
        #self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCELoss(reduce=False)
        #self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = nn.MultiLabelSoftMarginLoss()
        #self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass')
        self.confusion_matrix = None
        self.test_confusion_matrix = None
        #self.test_confusion_matrix = np.zeros((num_classes, num_classes))
        
        self.jump_predictions = []
        
    def single_label(self, y_new):
        # Make y_new only contain 0 or 1, where 0 is Background and 1 is the other classes
        y_single = torch.zeros((len(y_new), 1)).to(self.device)
        for i in range(len(y_new)):
            for j in range(len(y_new[i])):
                if y_new[i][j] != 0:
                    y_single[i] = 1
                else:
                    y_single[i] = 0
        return y_single
                    
    def multi_label_threshold(self, y_hat):
        # Take y_hat and set all values below 0.5 to 0 and all values above 0.5 to 1
        # If none of the values are above 0.5, set the highest value to 1
        predicted = y_hat.clone().detach().type(torch.FloatTensor).to(self.device)
        highest = 0
        highest_index = 0
        found_one = False
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                #print(predicted[i][j])
                if predicted[i][j] < 0.3:
                    predicted[i][j] = 0
                    if predicted[i][j] > highest:
                        highest = predicted[i][j]
                        highest_index = j
                else:
                    predicted[i][j] = 1
                    found_one = True
            if not found_one:
                # We then assume it is a jump
                #predicted[i][1] = 1
                pass
                #predicted[i][highest_index] = 1
                #print(predicted[i])
            highest = 0
            highest_index = 0
            found_one = False
            
        return predicted
        
    def default_step_work(self, batch, batch_idx, loss_log, acc_log):
        x, y, start, end, filename = batch
        y_hat = self(x)
        y_new = y.clone().detach().type(torch.FloatTensor).to(self.device)
        
        # Make weights vector for BCELoss, should be 2 for jump and 1 for everything else
        weights = [2, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1]
        weights = torch.FloatTensor(weights).to(self.device)
        loss = self.criterion(y_hat, y_new)
        
        # Multiply loss with weights
        loss = loss * weights
        loss = torch.mean(loss)
        
        
        #print(loss)
        self.log(loss_log, loss)
        predicted = self.multi_label_threshold(y_hat)
        #_, predicted = torch.max(y_hat, 1)
        #print(predicted)
        #accuracy = (predicted == y_new).sum().item() / len(y_new)
        accuracy = accuracy_score(y_new.cpu().numpy(), predicted.cpu().numpy())
        self.log(acc_log, accuracy)
        return loss
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        #print(x.shape)
        x = x.view(-1, 256 * x.shape[2] * x.shape[3])
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        loss = self.default_step_work(batch, batch_idx, 'train_loss', 'train_acc')
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.default_step_work(batch, batch_idx, 'val_loss', 'val_acc')
    
    def test_step(self, batch, batch_idx):

        x, y, start, end, filename = batch
        y_hat = self(x)
        y_new = y.clone().detach().type(torch.FloatTensor).to(self.device)
        
        weights = [2, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 1, 1, 1]
        weights = torch.FloatTensor(weights).to(self.device)
        loss = self.criterion(y_hat, y_new)
        
        # Multiply loss with weights
        loss = loss * weights
        loss = torch.mean(loss)
        
        #print(loss)
        self.log('test_loss', loss)
        predicted = self.multi_label_threshold(y_hat)
        #_, predicted = torch.max(y_hat, 1)
        #accuracy = (predicted == y_new).sum().item() / len(y_new)
        accuracy = accuracy_score(y_new.cpu().numpy(), predicted.cpu().numpy())
        self.log('test_acc', accuracy)
        
        # F1 score
        f1 = f1_score(y_new.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        self.log('test_f1', f1)
        
        # Recall
        recall = recall_score(y_new.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        self.log('test_recall', recall)
        
        # Precision
        precision = precision_score(y_new.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        self.log('test_precision', precision)
        
        # If sample is a jump or prediction is a jump, store its filename, time start and stop and the prediction and truth label in list.
        for i in range(len(predicted)):
            if predicted[i][0] == 1 or y_new[i][0] == 1:
                self.jump_predictions.append([filename[i], start[i], end[i], predicted[i].cpu().numpy(), y_new[i].cpu().numpy(), x[i].cpu().numpy()])

        
        
        self.confusion_matrix = multilabel_confusion_matrix(y_new.cpu().numpy(), predicted.cpu().numpy())
        if self.test_confusion_matrix is None:
            self.test_confusion_matrix = self.confusion_matrix
        else:
            self.test_confusion_matrix += self.confusion_matrix


        
    def on_test_epoch_end(self):
        #plot_confusion_matrix(self.test_confusion_matrix, self.classes, filename='confusion_matrix.png')
        plot_multilabel_confusion_matrix(self.test_confusion_matrix, self.classes, filename='CNN/multilabel_confusion_matrix_'+self.spectrogram_type+'.png')
        
        # Plot the confusion matrix just for the JUMP class
        utils.plot_binary_confusion_matrix(self.test_confusion_matrix[0], "JUMP", filename='CNN/confusion_matrix_jump_'+self.spectrogram_type+'.png')
        
        # print out the jump_predictions in a nice way
        print("Filename, Start, End, Prediction, Truth")
        plt.figure(figsize=(20, 10))
        for i in range(len(self.jump_predictions)):
            #print(self.jump_predictions[i][5].shape)
            plt.imshow(self.jump_predictions[i][5].squeeze(0))
            # Find out whether its TP, FP, TN or FN
            if self.jump_predictions[i][3][0] == 1 and self.jump_predictions[i][4][0] == 1:
                plt.title("True Positive for file :" + self.jump_predictions[i][0].split("\\")[1] + "\n" + " at time " + str(self.jump_predictions[i][1]) + " to " + str(self.jump_predictions[i][2]))
            elif self.jump_predictions[i][3][0] == 1 and self.jump_predictions[i][4][0] == 0:
                plt.title("False Positive for file :" + self.jump_predictions[i][0].split("\\")[1] + "\n" + " at time " + str(self.jump_predictions[i][1]) + " to " + str(self.jump_predictions[i][2]))
            elif self.jump_predictions[i][3][0] == 0 and self.jump_predictions[i][4][0] == 1:
                plt.title("False Negative for file :" + self.jump_predictions[i][0].split("\\")[1] + "\n" + " at time " + str(self.jump_predictions[i][1]) + " to " + str(self.jump_predictions[i][2]))
            elif self.jump_predictions[i][3][0] == 0 and self.jump_predictions[i][4][0] == 0:
                plt.title("True Negative for file :" + self.jump_predictions[i][0].split("\\")[1]+ "\n" + " at time " + str(self.jump_predictions[i][1]) + " to " + str(self.jump_predictions[i][2]))
            plt.savefig("graphs/" + self.jump_predictions[i][0].split("\\")[1]+".png", bbox_inches='tight')
        #plt.show()
        #print(self.jump_predictions)
        
        #print(self.test_confusion_matrix)
        return self.test_confusion_matrix
        
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch
        y_hat = self(x)

        predicted = self.multi_label_threshold(y_hat)
        #_, predicted = torch.max(y_hat, 1)
        return predicted

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer