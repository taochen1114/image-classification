import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from models import ClassificationModel
from losses import LossFunction
from config import get_config

from dataloader import get_loader
from tools import adjust_learning_rate, save_checkpoint
from tools import AverageMeter, EarlyStopping
from progress.bar import Bar
import time

def train(train_loader, model, criterion, optimizer):
    # Switch to train mode
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (_, inputs, targets) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        # Compute output
        outputs = model(inputs)          
        loss = criterion(outputs=outputs, targets=targets)
        
        # Record loss
        losses.update(loss.item(), inputs.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
       
        # Plot progress
        bar.suffix  = '({}/{}) Data: {:.3f}s | Batch: {:.3f}s | Total: {} | ETA: {} | Loss: {:.6f}'.format(
                        batch_idx + 1, 
                        len(train_loader), 
                        data_time.avg, 
                        batch_time.avg, 
                        bar.elapsed_td, 
                        bar.eta_td, 
                        losses.avg,
                        )
        bar.next()

    bar.finish() 
        
    return losses.avg

def validation(val_loader, model, criterion):
    # Switch to evaluate mode
    model.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    with torch.no_grad():
        bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (_, inputs, targets) in enumerate(val_loader):
            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs=outputs, targets=targets)

            # Record loss
            losses.update(loss.item(), inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            # Plot progress
            bar.suffix  = '({}/{}) Data: {:.3f}s | Batch: {:.3f}s | Total: {} | ETA: {} | Loss: {:.6f}'.format(
                            batch_idx + 1, 
                            len(val_loader), 
                            data_time.avg, 
                            batch_time.avg, 
                            bar.elapsed_td, 
                            bar.eta_td, 
                            losses.avg,
                            )
            bar.next()

        bar.finish() 

    return losses.avg


def main(args, state):

    torch.manual_seed(random.randint(1, 1000))

    train_loader, val_loader = get_loader(args=args)

    model = ClassificationModel(args.arch, args.num_classes, args.pretrain)
    # print(model)

    criterion = LossFunction(args.loss)
    print ('LossFunction: {} is ready!'.format(args.loss))
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, eps=1e-4)

    start_epoch = 0
    best_loss = 1000000

    print ('=====Start training and validation=====')
    early_stopping = EarlyStopping(patience=args.early_stop)

    for epoch in range(args.epochs):
        # Update learning rate
        optimizer = adjust_learning_rate(state, optimizer, epoch)
        
        print(f"Epoch: {epoch+1} | {args.epochs}, Learning Rate: {state['lr']}")
        
        train_loss = train(train_loader, model, criterion, optimizer)
        val_loss = validation(val_loader, model, criterion)

        
        # Save model
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': val_loss,
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
                'lr': state['lr'],
            }, is_best, checkpoint=args.checkpoint)
        
        early_stopping(val_loss, model)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return

if __name__ == "__main__":
    
    # Parse config arguments
    args, unparsed = get_config()
    state = {k: v for k, v in args._get_kwargs()}
    df_train = pd.read_csv(args.train)
    df_val = pd.read_csv(args.val)
        
    print(f"Model architecture: {args.arch}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Training csv: {args.train}")
    print(f"Train images: {df_train.shape[0]}")
    print(f"Validation csv: {args.val}")
    print(f"Validation images: {df_val.shape[0]}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Loss type: {args.loss}")
    print(f"Load pretrain weights: {args.pretrain}")
    print(f"Save checkpoint path: {args.checkpoint}")
    
    # Create logger file
    os.makedirs(args.checkpoint, exist_ok=True)
  
    main(args, state)
    