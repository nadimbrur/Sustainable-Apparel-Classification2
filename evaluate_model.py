import argparse
import pandas as pd
import numpy as np
import torchvision

import torch
import torch.nn as nn
from torch.autograd import Variable

import custom_dataset
import build_model
import average_meter

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='PyTorch Evaluting')

parser.add_argument('data', metavar='DIR',help='path to dataset')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='Batch-size ')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')


def main():


    """Taking Argument """
    args = parser.parse_args()
    path=args.data
    batch_size = args.batch_size
    learning_rate = args.lr

    """ Read testing CSV and select cuda"""
    test_csv = pd.read_csv(path,sep=',')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





    """ Create dataset and data loader"""
    test_set = custom_dataset.CustomDataset(test_csv, transform=transforms.Compose([transforms.ToTensor(),
                                                                                    transforms.Normalize(mean=[.5],
                                                                                                         std=[.5])]))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



    """ Create model, optimizer and loss"""
    model = build_model.BuildModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    losses1 = average_meter.AverageMeter()
    top1_val = average_meter.AverageMeter()
    top5_val = average_meter.AverageMeter()



    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        """ Initialize variable"""
        label_pre = []
        label_act = []
        total = 0
        correct = 0


        """  Load checkpoint on the model from path"""

        temp='./checkpoint/eval_model_best.pth.tar'
        checkpoint = torch.load(temp, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model=model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])

        """ Iterate throught test loader """
        for i, (input, target) in enumerate(test_loader):
            if device !="cpu":
                input = input.cuda(device, non_blocking=True)
            target = target.type(torch.LongTensor)
            target = target.cuda(device, non_blocking=True)

            # compute output
            input = Variable(input.view(-1, 1, 28, 28))
            output = model(input)

            # compute loss
            loss = criterion(output, target)
            predictions = torch.max(output, 1)[1].to(device)


            # compute total correct prediction and total labels
            correct += (predictions == target).sum()
            total += len(target)

            # compute accuracy
            accuracy_h = correct * 100 / total
            label_act+=target
            label_pre+=predictions
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


        #Print total accuracy
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
             .format(top1=top1_val, top5=top5_val))
        print("Total accuracy is: ",accuracy_h.item())

        #compute confusion matrix and each class classification report
        label_act=torch.tensor(label_act).detach().cpu().numpy()
        label_pre = torch.tensor(label_pre).detach().cpu().numpy()
        name=['T-shirt/top', 'Trouser', 'Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        confusion_matrix= sklearn.metrics.confusion_matrix(label_act,label_pre)
        print(confusion_matrix)
        report_matrix = sklearn.metrics.classification_report(label_act, label_pre,target_names=name,digits=4)
        print(report_matrix)



        """  Write Model architecture details and classification report along with accuracy and loss"""
        """Count number of parameter of the model"""
        total_params = sum(p.numel() for p in model.parameters())
        # print(f'')
        try:
           with open('output.txt', 'w') as file:
               file.write("The model architecute are given below:\n\n",)
               file.write(str(model))
               file.write('\n\n\n\nTotal number of parameters in the model:'+str(total_params))
               file.write("\n\n\nThe loss is:")
               file.write(str(losses1.avg))
               file.write("\n\n\nThe accuracy of each class are give below:\n\n")
               file.write(report_matrix)
               if(acc1>=75):
                   file.write("\n\n  The accuracy is very good for classifiying each class and overall accuracy is: ")
                   file.write(str(accuracy_h.item()))


        except Exception as err:
            print(err)
        finally:
            print("Please check generated 'output.txt' for more details")
            # Clean exit, no matter if there was an exception or not
            exit()

def accuracy_measure(output, target, topk=(1,)):
   """ computes the accuracy over the k top predictions for the specified values"""
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
if __name__ == '__main__':
   main()
