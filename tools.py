import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import seaborn as sn

from shutil import copyfile

from itertools import cycle
from config import get_config
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize

args, unparsed = get_config()

def adjust_learning_rate(state, optimizer, epoch):
    if epoch in state['schedule']:
        state['lr'] *= state['gamma']
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
    return optimizer


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpoint='model_ckpt', 
                    filename='model_ckpt.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def print_classification_report(gt_list, pred_list):
    y_true = np.array(gt_list).astype(int)
    y_pred = np.array(pred_list).astype(int)

    # count precision_score, recall_score, f1_score
    # print('count metrics globally:')
    # print('precision:', precision_score(y_true, y_pred, average='micro'))
    # print('recall:', recall_score(y_true, y_pred, average='micro'))
    # print('f1_score:', f1_score(y_true, y_pred, average='micro'))

    print('count metrics for each label and count average:')
    print('cls avg precision:', precision_score(y_true, y_pred, average='macro'))
    print('cls avg recall:', recall_score(y_true, y_pred, average='macro'))
    print('cls avg f1_score:', f1_score(y_true, y_pred, average='macro'))

    print(classification_report(y_true, y_pred))


def draw_cm(y_true, y_pred, save_dir, class_def):
    print ('=========Draw confusion matrix=========')
    # load label name for classes
    with open(class_def) as f:
        classes = f.readlines()
        classes = [c.rstrip('\n') for c in classes] # classes = ["Cat", "Dog"]


    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    df_cm = pd.DataFrame(cm/np.sum(cm) *2, index = [i for i in classes],
                         columns = [i for i in classes])
    # plt.figure(figsize = (12,7))
    plt.figure()
    plt.title("Confusion Matrix")
    sn.heatmap(df_cm, annot=True)
    plt.ylabel("Groundtruth")
    plt.xlabel("Prediction")
    plt.savefig(os.path.join(save_dir, "Confusion_matrix.png"))

    print(f"==> Save confusion matrix to: {os.path.join(save_dir, 'Confusion_matrix.png')}")
    plt.close()


def draw_roc_curve(gt_onehot_list, probs_list, save_dir, class_def, num_classes):
    print ('==============ROC Analysis=============')
    # load label name for classes
    with open(class_def) as f:
        classes = f.readlines()
        classes = [c.rstrip('\n') for c in classes] # classes = ["Cat", "Dog"]


    gt = np.array(gt_onehot_list)
    y_score = np.array(probs_list)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    lw = 2
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(gt[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(gt.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig_roc = plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], 
        label="micro-average ROC curve (area = {0:0.3f})".format(roc_auc["micro"]),
        color="deeppink", linestyle=":", linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"], 
        label="macro-average ROC curve (area = {0:0.3f})".format(roc_auc["macro"]),
        color="navy", linestyle=":", linewidth=4)

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "maroon"])
    for i, (c, color) in enumerate(zip(classes, colors)):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, 
            label="ROC curve of class {0} (area = {1:0.3f})".format(c, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic Curve")
    plt.legend(loc="lower right")
    # plt.show()
    fig_roc.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=fig_roc.dpi, bbox_inches="tight")
    print(f"==> Save roc curve to: {os.path.join(save_dir, 'roc_curve.png')}") 
    