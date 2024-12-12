# @Time    : 2024/06/01 21:09
# @Author  : Yang Xionghui
# @File    : test.py
# @Software: PyCharm

import torch
import torch.nn as nn
import os, sys, pdb
import argparse
import torchvision
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
import pandas as pd
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
# from Cpyramid_vig import Cpvig_ti_224_gelu
def build_cof_mat(num, poly_num=2):
    assert num > 2, "num should bigger than 2"
    cof_mat = np.zeros((num, num), dtype=np.float32)
    for i in range(0, num):
        for j in range(0, num):
            cof_mat[i, j] = np.power(np.absolute(i-j), poly_num)

    return cof_mat
def ordinal_mse(confusion_matrix, poly_num=2):
    category_num = confusion_matrix.shape[0]
    cof_mat = build_cof_mat(category_num, poly_num=poly_num)

    err_mat = np.multiply(confusion_matrix, cof_mat)
    mse_val = np.sum(err_mat) * 1.0 / np.sum(confusion_matrix)

    return mse_val

class ConfusionMatrix(object):
    def __init__(self, args,labels):
        self.matrix = None # 初始化混淆矩阵，元素都为0
        self.args = args
        self.labels = labels  # 类别标签
        # self.save_txt = os.path.join(os.path.dirname(os.path.dirname(args.best_model_path)), "1.txt")
        self.save_txt = os.path.join(os.path.join(args.log_directory, args.log_name), "1.txt")
        self.df = None #用于打印计算的各项指标
        self.accuracy = 0 #当前的准确率
        self.kappa = 0 #当前的kappa值
    def summary(self,labels_all,preds_all):  # 计算指标函数
        # 假设labels_all是真实标签，preds_all是预测结果
        self.matrix = confusion_matrix(labels_all, preds_all)
        report = classification_report(labels_all, preds_all, output_dict=True,zero_division=0)
        self.kappa = cohen_kappa_score(labels_all, preds_all)
        self.df = pd.DataFrame(report).transpose()
        self.accuracy = self.df.loc['accuracy', 'precision']
        mse = ordinal_mse(self.matrix, poly_num=1)
        return self.matrix,mse

    def plot(self,normalize=False,title=None,cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        #显示DataFrame
        # print(self.df)
        #self.accuracy = self.df.loc['accuracy', 'precision']
        # print("the model accuracy is : ", self.accuracy)
        print("the model kappa is: ", self.kappa)
        f = open(self.save_txt, mode="w")
        f.write(str(self.df.to_string()) + '\n\n\n')


        classes=self.labels
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
                
        # Compute confusion matrix
        cm = self.matrix
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.title('Confusion matrix (acc=' + str(self.accuracy) + ')')
        plt.savefig(os.path.join(os.path.join(self.args.log_directory, self.args.log_name), 'matrix_{}.png'.format(self.accuracy)), format='png')
        plt.close()
