import numpy as np
import pandas as pd
import argparse
import os
import shutil
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import custom_dataset
import build_model
import average_meter

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='PyTorch CNN Training')



parser.add_argument('traindata', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--epochs', default=20,type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='Batch-size ')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')





def accuracy_measure(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    """Taking Argument """
    args = parser.parse_args()
    train_path=args.traindata
    batch_size=args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr


    """ Read testing CSV and select cuda"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_csv = pd.read_csv(train_path)
    p=(int)(len(train_csv)*.2)
    test_csv,train_csv= train_csv[:p],train_csv[p:]
    # print(len(train_csv),len(test_csv))

    # print(h)


    """ Create dataset and data loader"""

    # Transform data into Tensor that has and other
    train_set =custom_dataset.CustomDataset(train_csv, transform=transforms.Compose([
          transforms.ToPILImage(),
           transforms.RandomHorizontalFlip(),
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           transforms.RandomErasing(p=0.5),
           transforms.ColorJitter(0.4),
           transforms.Normalize(mean=[.5], std=[.5])

        ]))
    test_set = custom_dataset.CustomDataset(test_csv, transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[.5], std=[.5])
                                                          ]))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)




    """ Create model, optimizer and loss"""
    model = build_model.BuildModel()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)




    # Initialize variable
    predictions_list = []
    labels_list = []
    data_time = average_meter.AverageMeter()
    losses =  average_meter.AverageMeter()
    top1 =  average_meter.AverageMeter()
    top5 =  average_meter.AverageMeter()
    max_acc = 0
    el = 0
    train_loss=[]
    acc_t_loss=[]
    acc_v_loss=[]
    val_loss=[]


    """ Start train depends on epochs"""
    for epoch in range(num_epochs):

        # switch to train mode
        model.train()

        running_loss=0
        """ Iterate Throught train loader """
        for i,(images, labels) in enumerate (train_loader):

            images, labels = images.to(device), labels.to(device)
            input = Variable(images.view(-1, 1, 28, 28))
            labels = Variable(labels)


            # Forward pass and  compute output
            outputs = model(input)
            labels=labels.to(torch.int64)
            loss = criterion(outputs, labels)
            running_loss = running_loss+ (loss.item() * input.size(0))
            optimizer.zero_grad()
            loss.backward()




            # Optimizing the parameters
            optimizer.step()


            # measure accuracy and record loss
            acc1, acc5 = accuracy_measure(outputs, labels, topk=(1, 4))



            # compute average loss and accuracy rank
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))


            """  Print after 100 iterations"""
            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader),
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        scheduler.step()
        train_loss.append(running_loss / len(train_set))

        acc_t_loss.append(losses.avg)

        #Initializing variable for evaluating
        losses1 =  average_meter.AverageMeter()
        top1_val =  average_meter.AverageMeter()
        top5_val =  average_meter.AverageMeter()
        # switch to evaluate mode
        model.eval()

        #To parameter will update
        with torch.no_grad():
            total = 0
            correct = 0
            val_running_loss=0
            for i, (input, target) in enumerate(test_loader):
                if device != "cpu":
                    input = input.cuda(device, non_blocking=True)
                target = target.type(torch.LongTensor)
                target = target.cuda(device, non_blocking=True)

                # compute output
                input=Variable(input.view(-1, 1, 28, 28))
                output = model(input)
                # compute loss
                loss = criterion(output, target)
                val_running_loss=val_running_loss+(loss*input.size(0))

                # compute total correct prediction and total labels
                labels_list.append(target)
                predictions = torch.max(output, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == target).sum()
                total += len(target)

                # compute accuracy
                accuracy_h = correct * 100 / total
                acc1, acc5 = accuracy_measure(output, target, topk=(1, 4))
                # compute average loss and accuracy rank
                losses1.update(loss.item(), input.size(0))
                top1_val.update(acc1[0], input.size(0))
                top5_val.update(acc5[0], input.size(0))

                """  Print after 100 iteration"""
                if i % 100 == 0:
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(test_loader), loss=losses1,
                        top1=top1_val, top5=top5_val))

            # Print total accuracy
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1_val, top5=top5_val))
            print("Now is",accuracy_h)


            #Update accuracy if it is current best
            is_best = accuracy_h > max_acc
            if max_acc<accuracy_h:
                max_acc=accuracy_h
                el=epoch

            """Save the last check point and also save the best checkpoint"""
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': accuracy_h,
                'optimizer': optimizer.state_dict(),
            }, is_best,epoch)
            print("Still the best is:", max_acc,"Corresponding Epoch is:",el)
        val_loss.append(val_running_loss/len(test_set))
        acc_v_loss.append(losses1.avg)



    """Plot validation loss and training loss"""
    # Plotting the values from arrays
    plt.figure(figsize=(8, 6))  # Optional: Set the figure size
    acc_v_loss = torch.tensor(acc_v_loss).detach().cpu().numpy()
    acc_t_loss = torch.tensor(acc_t_loss).detach().cpu().numpy()
    plt.plot(acc_t_loss, label='Train_loss', marker='o')
    plt.plot(acc_v_loss, label='Validation_loss', marker='x')
    plt.title('Train and validation loss')  # Plot title
    plt.legend()  # Show legend to distinguish between
    plt.grid(True)  # Show grid in the plot (optional)
    plt.xlabel('Epoch')  # X-axis label
    plt.ylabel('Loss')  # Y-axis label
    plt.show()


def save_checkpoint(state, is_best, epoch,filename='checkpoint.pth.tar'):
    x=r'./checkpoint'
    filename=str(epoch)+filename
    # print(filename, is_best)
    x=os.path.join(x,filename)
    torch.save(state, x)

    """If best accuracy, copy to create model_best"""
    if is_best:
        shutil.copyfile(x, './checkpoint/model_best.pth.tar')

if __name__ == '__main__':
    main()
