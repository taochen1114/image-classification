import os
import pandas as pd
import numpy as np

import torch 
from models import ClassificationModel 
from losses import LossFunction
from config import get_config

from dataloader import get_loader
from tools import AverageMeter
from progress.bar import Bar
import time

from tools import draw_cm, print_classification_report, draw_roc_curve

def inference(args):
    test_loader = get_loader(args=args)

    model = ClassificationModel(args.arch, args.num_classes, False)

    criterion = LossFunction(args.loss)
    print ('LossFunction: {} is ready!'.format(args.loss))

    # load best model 
    model_path = os.path.join(args.checkpoint, args.model_file) # test_modelfname default: model_best.pth.tar
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # for inference report measuring
    pred_list, gt_list, probs_list, gt_onehot_list = [], [], [], []
    filename_list, correct_list = [], []

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end = time.time()

    print ('=====Start testing =====')
    with torch.no_grad():
        bar = Bar('Processing', max=len(test_loader))
        for batch_idx, (filenames, inputs, targets) in enumerate(test_loader):
            # Compute output
            outputs = model(inputs).float()
            outputs = torch.sigmoid(outputs)

            y_true = torch.argmax(targets.data, dim=1)
            
            # Measure accuracy and record loss
            _, y_pred = torch.max(outputs.data, 1)
            probs_list += outputs.data.tolist()   # add to probability value list
            pred_list += y_pred.tolist()      # predict label value list
            gt_list += y_true.tolist() # ground truth label value list
            gt_onehot_list += targets.data.tolist() # ground truth one-hot label value list
            filename_list += list(filenames)

            correct_list += y_pred.eq(y_true).tolist()

            loss = criterion(outputs=outputs, targets=targets)

            # Record loss
            losses.update(loss.item(), inputs.size(0))

            # Record accuracy
            correct = (y_pred == y_true).float().sum()
            accuracy.update(correct/inputs.size(0), inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            # Plot progress
            bar.suffix  = '({}/{}) Data: {:.3f}s | Batch: {:.3f}s | Total: {} | ETA: {} | Loss: {:.6f} | Acc: {:.6f}'.format(
                            batch_idx + 1, 
                            len(test_loader), 
                            data_time.avg, 
                            batch_time.avg, 
                            bar.elapsed_td, 
                            bar.eta_td, 
                            losses.avg,
                            accuracy.avg,
                            )
            bar.next()

        bar.finish() 

        print ('Average top1 accuracy: {0:.6f}'.format(accuracy.avg))

    # draw_confusion_matrix(gt_list, pred_list, args.checkpoint)
    print_classification_report(gt_list, pred_list)

    draw_cm(gt_list, pred_list, args.checkpoint, args.class_def)
    draw_roc_curve(gt_onehot_list, probs_list, args.checkpoint, args.class_def, args.num_classes)

    return 


if __name__ == "__main__":
    
    # Parse config arguments
    args, unparsed = get_config()
    # state = {k: v for k, v in args._get_kwargs()}
    df_test = pd.read_csv(args.test)
        
    print(f"Model architecture: {args.arch}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Testing csv: {args.test}")
    print(f"Testing images: {df_test.shape[0]}")
    print(f"Batch size: {args.batch}")
    print(f"Loss type: {args.loss}")
    print(f"Load checkpoint path: {args.checkpoint}")
    
    inference(args)
    