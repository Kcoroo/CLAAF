import sys
import random
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import SorcenDatasets
import model
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sup_ctl_loss import SupConLoss
tqdm.disable = True

def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=80, help='batch_size')
parser.add_argument('--max_epochs', type=int, default=50, help='max_epochs')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='device')
parser.add_argument('--model_name', type=str, default='./models/sorcen_model.pth', help='model path')
parser.add_argument('--lr', type=float, default=5e-5, help='learning_rate')
parser.add_argument('--classes', type=int, default=2, help='classes')
parser.add_argument('--alpha', type=float, default=0.3, help='alpha to ctl')
args = parser.parse_args()

set_random_seed(99003)

def train_model(opt, model):
    model = model.to(opt.device)
    plot_train_loss = []
    plot_train_acc = []
    plot_val_loss = []
    plot_val_acc = []

    best_acc = 0
    length_all = len(train_data)
    length_all_val = len(test_data)

    for epoch in range(opt.max_epochs):
        temp_train_loss = 0
        temp_train_acc = 0
        temp_val_loss = 0
        temp_val_acc = 0

        # train
        model.train()

        for x1_1, x1_2, x1_3, x2, y in tqdm(train_dataloader):
            x1_1, x1_2, x1_3, x2, y = x1_1.to(opt.device), x1_2.to(opt.device), x1_3.to(opt.device), x2.to(opt.device), y.to(opt.device)

            optimizer.zero_grad()
            cls_out, cls_y = model(x1_1, x1_2, x1_3, x2)
            loss_cls = loss_func(cls_y, y)
            loss_ctl = loss_ctl_func(cls_out, y)
            loss = loss_cls + loss_ctl * opt.alpha
            # loss = loss_cls
            loss.backward()
            optimizer.step()
            pred = torch.max(cls_y, dim=1)[1]
            true = y
            acc = float(torch.eq(true.cpu(), pred.cpu()).sum() / len(x2))
            temp_train_loss += loss.item() * len(x2)
            temp_train_acc += acc * len(x2)

        # val
        test_steps = len(test_dataloader)
        res = 0.0

        model.eval()
        with torch.no_grad():
            for x1_1, x1_2, x1_3, x2, y in tqdm(test_dataloader):
                optimizer.zero_grad()
                x1_1, x1_2, x1_3, x2, y = x1_1.to(opt.device), x1_2.to(opt.device), x1_3.to(opt.device), x2.to(opt.device), y.to(opt.device)
                cls_out, y_1 = model(x1_1, x1_2, x1_3, x2)

                loss_cls = loss_func(y_1, y)
                loss_ctl = loss_ctl_func(cls_out, y)
                loss_10 = loss_cls + loss_ctl * opt.alpha
                # loss_10 = loss_cls

                # Loss_all = loss_func((y_2 + y_1) / 2, y)
                pred = torch.max(y_1, dim=1)[1]
                # true = torch.max(y, dim=1)[1]
                true = y
                acc = float(torch.eq(true.cpu(), pred.cpu()).sum() / len(x2))
                # res += loss_10.item()
                temp_val_loss += loss_10.item() * len(x2)
                temp_val_acc += acc * len(x2)
            scheduler.step((temp_val_loss / length_all_val))

        print(f'|| {epoch + 1} / {opt.max_epochs} ||'
              f'|| training_loss {temp_train_loss / length_all:.4f} ||'
              f'|| verification_loss {temp_val_loss / length_all_val:.4f} ||')
        print(f'|| training_accuracy {temp_train_acc / length_all:.4f} ||'
              f'|| verification_accuracy {temp_val_acc / length_all_val:.4f} ||')

        plot_train_loss.append(temp_train_loss / length_all)
        plot_train_acc.append(temp_train_acc / length_all)

        plot_val_loss.append(temp_val_loss / length_all_val)
        plot_val_acc.append(temp_val_acc / length_all_val)

        if plot_val_acc[-1] > best_acc:
            best_acc = plot_val_acc[-1]
            if not os.path.exists('./models/'):
                os.mkdir('./models/')
            torch.save(model, opt.model_name)


def confusion_matrix_(preds, labels, matrix, pred_ed=False):
    if not pred_ed:
        preds = torch.argmax(preds, 1)
        # labels = torch.argmax(labels, 1)
    for p, t in zip(preds, labels):
        matrix[p, t] += 1
    return matrix

def get_test_result(y_true, y_pred):
    confusion_mat = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, recall, f1, precision, confusion_mat

def test_model(opt, test_dataloader, num_classes=4):
    model = torch.load(opt.model_name, map_location=opt.device)
    model.eval()
    temp_test_acc = 0
    length = len(test_data)
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        conf_mat = torch.zeros(num_classes, num_classes)
        for x1_1, x1_2, x1_3, x2, y in tqdm(test_dataloader):
            x1_1, x1_2, x1_3, x2, y = x1_1.to(opt.device), x1_2.to(opt.device), x1_3.to(opt.device), x2.to(opt.device), y.to(opt.device)
            cls_out, y_1 = model(x1_1, x1_2, x1_3, x2)
            pred = torch.max(y_1 , dim=1)[1]
            # true = torch.max(y, dim=1)[1]
            true = y
            acc = float(torch.eq(true.cpu(), pred.cpu()).sum() / len(x2))
            y_pred_list.extend(pred.cpu())
            y_true_list.extend(true.cpu())
            conf_mat = confusion_matrix_(y_1, y, conf_mat)
            temp_test_acc += acc * len(x2)
        print(f'|| test_accuracy {temp_test_acc / length:.4f} ||')
        accuracy, recall, f1, precision, confusion_mat = get_test_result(y_true_list, y_pred_list)
        print(f'|| test accuracy {accuracy} |||| recall {recall} |||| f1 {f1} |||| precision {precision} \n confusion_mat : \n {confusion_mat}')


if __name__ == '__main__':
    
    classes = args.classes
    print(args)
    train_data = SorcenDatasets('train')
    val_data = SorcenDatasets('valid')
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=args.batch_size)

    test_data = SorcenDatasets('test')
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    print(f'|| train_data || {len(train_data)}'
        f'|| val_data || {len(val_data)}')
    loss_func = nn.CrossEntropyLoss()
    loss_ctl_func = SupConLoss(temperature=0.07)

    model = model.Model()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
    print('-----train model-----')
    train_model(args, model)
    test_model(args, test_dataloader, num_classes=classes)