import os
import math
from math import isnan
import re
import pickle
import gensim
import numpy as np
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from scipy.special import expit
from torchviz import make_dot
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):
        self.train_accuracies = []
        self.valid_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.train_maes = []
        self.valid_maes = []
        self.test_maes = []
        self.train_f1_scores = []
        self.valid_f1_scores = []
        self.test_f1_scores = []
        
        
        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
        self.criterion = criterion = nn.L1Loss(reduction="mean")
    @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            #print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    
    def eval( self,folder, mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(folder+'/model_best_acc.std'))
                optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

            

        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                mean = 0.0
                std = 0.5

                gaussian_noise = torch.normal(mean=mean, std=std, size=v.shape)
                #v = v + 2.00*gaussian_noise
                gaussian_noise = torch.normal(mean=mean, std=std, size=t.shape)
                #t = t + 2.00*gaussian_noise
                gaussian_noise = torch.normal(mean=mean, std=std, size=a.shape)
                #a = a + 2.00*gaussian_noise
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)
                y_tilde = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                
                cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)
        mae = np.mean(np.abs(y_pred - y_true))
        f1 = f1_score((y_pred >= 0), (y_true >=0), average='weighted')
        
        if mode == "dev":
            self.valid_losses.append(eval_loss)
            self.valid_accuracies.append(accuracy)
            self.valid_maes.append(mae)
            self.valid_f1_scores.append(f1)
        elif mode == "test":
            self.test_losses.append(eval_loss)
            self.test_accuracies.append(accuracy)
            self.test_maes.append(mae)
            self.test_f1_scores.append(f1)
        if to_print:
            print(f"Eval {mode} loss: {round(eval_loss, 4)}, Accuracy: {round(accuracy, 4)}, MAE: {round(mae, 4)}, F1-score: {round(f1, 4)}")
            
            
        return eval_loss, accuracy, mae, f1

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Metric scheme adapted from:
        https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
        """


        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
            
            return accuracy_score(test_truth, test_preds)

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
            test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
            test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            corr = np.corrcoef(test_preds, test_truth)[0][1]
            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)
            
            pos_neg_f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
            
            # pos - neg
            binary_truth = (test_truth[non_zeros] > 0)
            binary_preds = (test_preds[non_zeros] > 0)

            if to_print:
                print("mae: ", mae)
                print("corr: ", corr)
                print("mult_acc: ", mult_a7)
                print("Classification Report (pos/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
                print("F1 (pos/neg) ", pos_neg_f_score)
            
            # non-neg - neg
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)
            non_neg_f_score = f1_score(binary_truth, binary_preds, average='weighted')
            if to_print:
                print("Classification Report (non-neg/neg) :")
                print(classification_report(binary_truth, binary_preds, digits=5))
                print("Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
                
                print("F1 (non-neg/neg) ", non_neg_f_score)
            
            return accuracy_score(binary_truth, binary_preds)





