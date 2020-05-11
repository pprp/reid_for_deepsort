import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
from torchvision import datasets

from eval import get_result
from models import build_model

matplotlib.use('Agg')

input_size = (128, 128)

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=10, type=int)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--model', type=str, default="resnet50_ibn_a")
parser.add_argument('--pretrained', action="store_true")
args = parser.parse_args()

# device
device = "cuda:{}".format(
    args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"

if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
root = args.data_dir
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "val")

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.3568, 0.3141, 0.2781],
                                     [0.1752, 0.1857, 0.1879])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.3568, 0.3141, 0.2781],
                                     [0.1752, 0.1857, 0.1879])
])
train_datasets = datasets.ImageFolder(train_dir, transform=transform_train)
test_datasets = datasets.ImageFolder(test_dir, transform=transform_test)

trainloader = torch.utils.data.DataLoader(train_datasets,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=4)

testloader = torch.utils.data.DataLoader(test_datasets,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=4)

num_classes = len(trainloader.dataset.classes)

##################
# net definition #
##################

start_epoch = 0
net = build_model(name=args.model, num_classes=num_classes,
                  pretrained=args.pretrained)

if args.resume:
    assert os.path.isfile(
        "./weights/best.pt"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/best.pt')
    checkpoint = torch.load("./weights/best.pt")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

net.to(device)

# loss and optimizer
criterion_model = torch.nn.CrossEntropyLoss()
optimizer_model = torch.optim.SGD(
    net.parameters(), args.lr)  # from 3e-4 to 3e-5
scheduler = optim.lr_scheduler.StepLR(  # best lr 1e-3
    optimizer_model, step_size=20, gamma=0.1)

best_acc = 0.


# train function for each epoch
def train(epoch):
    print('=' * 30, "Training", "=" * 30)
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        loss = criterion_model(outputs, labels)

        # backward
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()

        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        if (idx + 1) % interval == 0:
            end = time.time()
            print(
                "epoch:{:d}|step:{:03d}|time:{:03.2f}s|Loss:{:03.5f}|Acc:{:02.3f}%"
                .format(epoch, idx, end - start, training_loss / interval,
                        100. * correct / total))
            training_loss = 0.
            start = time.time()
    return train_loss / len(trainloader), 1. - correct / total


def test(epoch):
    global best_acc
    # net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion_model(outputs, labels)
            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print('=' * 30, "Testing", "=" * 30)

        end = time.time()
        print(
            "epoch:{:d}\t time:{:.2f}s\t Loss:{:.5f}\t Correct:{}/{}\t Acc:{:.3f}%"
            .format(epoch, end - start, test_loss / len(testloader), correct,
                    total, 100. * correct / total))

    # saving checkpoint
    acc = 100. * correct / total
    if not os.path.isdir('weights'):
        os.mkdir('weights')

    save_path = os.path.join("weights", args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/best.pt")

        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(checkpoint,
                   './weights/%s/%s_best.pt' % (args.model, args.model))
        torch.save(checkpoint,
                   './weights/%s/%s_last.pt' % (args.model, args.model))
    else:
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(checkpoint,
                   './weights/%s/%s_last.pt' % (args.model, args.model))
    # rank and mAP
    # net.eval()
    # TODO BUG
    # get_result(net, trainloader, testloader, train_datasets, test_datasets)

    return test_loss / len(testloader), 1. - correct / total


# plot figure
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")


if __name__ == '__main__':
    for epoch in range(start_epoch, start_epoch + 200):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        scheduler.step()
        # if epoch % 10 == 0:
        # os.system("python eval.py")
