import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import sys

from matplotlib import pyplot as plt

import mltools
from sklearn.metrics import accuracy_score
from config_resnet2 import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_dataset = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
train_loader = Data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
val_dataset = Data.TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
val_loader = Data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

# cross entropy loss
criterion = nn.CrossEntropyLoss()
# model optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# print(model)


def to_device(device):
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def train_torch(start_epoch=1):
    print('Start Training')
    print('Using lam_center {}, lam_encoder {}, feature dimension {}, version {}'.format(lam_center, lam_encoder,
                                                                                         feature_dim, version))

    if os.path.exists('models/' + version):
        shutil.rmtree('models/' + version)
        os.mkdir('models/' + version)
    else:
        os.mkdir('models/' + version)

    if not os.path.isdir('models/' + version + '/tsne/'):
        os.mkdir('models/' + version + '/tsne/')

    model_batch_path = './models/{}/model_{}d_{}.pkl'
    loss_txt_path = f'./models/{version}/loss.txt'

    loss_list_train = []
    accuracy_list_train = []

    loss_list_val = []
    accuracy_list_val = []

    to_device(device)

    model.train()
    with open(loss_txt_path, 'w') as f:
        resultlines = []
        for epoch in range(start_epoch, start_epoch + epoch_num):
            model.train()
            feature_output_train = []
            feature_label_train = []
            pred_labels_train, true_labels_train = [], []
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                labels = torch.max(labels.long(), 1)[1]

                feature_output_train.append(model.getSemantic(inputs).cpu().detach().numpy())
                feature_label_train.append(labels.cpu().detach().numpy())

                pred_labels_train.append(np.argmax(outputs.cpu().detach().numpy(), axis=1))
                true_labels_train.append(labels.cpu().detach().numpy())

                loss_cross_train = criterion(outputs, labels)
                loss_train = loss_cross_train
                loss_train.backward()
                optimizer.step()

            with torch.no_grad():
                pred_labels_val, true_labels_val = [], []
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    labels = torch.max(labels.long(), 1)[1]

                    pred_labels_val.append(np.argmax(outputs.cpu().detach().numpy(), axis=1))
                    true_labels_val.append(labels.cpu().detach().numpy())

                    loss_cross_val = criterion(outputs, labels)
                    loss_val = loss_cross_val

            accuracy_train = accuracy_score(np.hstack(true_labels_train), np.hstack(pred_labels_train))
            accuracy_list_train.append(round(accuracy_train, 3))

            accuracy_val = accuracy_score(np.hstack(true_labels_val), np.hstack(pred_labels_val))
            accuracy_list_val.append(round(accuracy_val, 3))

            print('[%d/%d] train loss: %.3f, cross loss: %.3f, accuracy: %.3f\n         val loss: %.3f, cross loss: '
                  '%.3f, accuracy: %.3f' % (
                      epoch, start_epoch + epoch_num - 1, loss_train.item(), loss_cross_train.item(),
                       accuracy_train, loss_val.item(),
                      loss_cross_val.item(),   accuracy_val))
            resultlines.append('[%d/%d] train loss: %.3f, cross loss: %.3f,  accuracy: %.3f\n         val loss: %.3f, '
                               'cross loss: %.3f, accuracy: %.3f\n' % (
                    epoch, start_epoch + epoch_num - 1, loss_train.item(), loss_cross_train.item(),
                     accuracy_train, loss_val.item(),
                    loss_cross_val.item(),  accuracy_val))

            loss_list_train.append(np.round(loss_train.item(), 3))
            loss_list_val.append(np.round(loss_val.item(), 3))

            # mltools.tsne(feature_output, feature_label, train_mods, f'./models/{version}/tsne/tsne_{epoch}.jpg')

            if epoch >= 25 and epoch % 25 == 0:
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': start_epoch + epoch
                }
                torch.save(state, model_batch_path.format(version, feature_dim, epoch))
                mltools.tsne(feature_output_train, feature_label_train, train_mods,
                             f'./models/{version}/tsne/tsne_{epoch}.jpg')

        f.writelines(resultlines)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': start_epoch + epoch_num}
    torch.save(state, model_path)
    print('Finished Training')

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    plt.plot(range(1, epoch_num + 1), loss_list_train, color='blue', linewidth=1, linestyle="solid", label="train loss")
    plt.plot(range(1, epoch_num + 1), loss_list_val, color='red', linewidth=1, linestyle="solid", label="val loss")
    plt.legend(loc=0)
    plt.grid()
    plt.title('Loss curve')
    plt.savefig(f'./models/{version}/loss.jpg')
    # plt.show()
    plt.close()
    np.save(f'./models/{version}/loss_train.npy', loss_list_train)
    np.save(f'./models/{version}/loss_val.npy', loss_list_val)

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('accuracy')  # y轴标签

    plt.plot(range(1, epoch_num + 1), accuracy_list_train, color='blue', linewidth=1, linestyle="solid",
             label="train accuracy")
    plt.plot(range(1, epoch_num + 1), accuracy_list_val, color='red', linewidth=1, linestyle="solid",
             label="val accuracy")
    plt.legend(loc=0)
    plt.grid()
    plt.title('Accuracy curve')
    plt.savefig(f'./models/{version}/accuracy.jpg')
    # plt.show()
    plt.close()
    np.save(f'./models/{version}/accuracy_train.npy', accuracy_list_train)
    np.save(f'./models/{version}/accuracy_val.npy', accuracy_list_val)


if __name__ == '__main__':
    start_epoch = 1
    if len(sys.argv) > 1 and sys.argv[1] == '-r':
        print('Resuming model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

        print('Resume training from epoch {}'.format(start_epoch))

    train_torch(start_epoch)
