import torch
import torch.backends.cudnn as cudnn
import torchvision

import argparse
import os

from model import Net
from osnet import *

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir", default='data', type=str)
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--gpu-id", default=0, type=int)
args = parser.parse_args()

# device
device = "cuda:{}".format(
    args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
root = args.data_dir
query_dir = os.path.join(root, "query")
gallery_dir = os.path.join(root, "gallery")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
])
queryloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    query_dir, transform=transform),
                                          batch_size=64,
                                          shuffle=True)
galleryloader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
    gallery_dir, transform=transform),
                                            batch_size=64,
                                            shuffle=True)

num_classes = len(queryloader.dataset.classes)

# net definition
net = osnet_x1_0(num_classes=num_classes,reid=True)
assert os.path.isfile(
    "./checkpoint/best.pt"), "Error: no checkpoint file found!"
print('Loading from checkpoint/best.pt')
checkpoint = torch.load("./checkpoint/best.pt")
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict)
net.eval()
net.to(device)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
gallery_features = torch.tensor([]).float()
gallery_labels = torch.tensor([]).long()

with torch.no_grad():
    for idx, (inputs, labels) in enumerate(queryloader):
        print(idx, labels)
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))

    for idx, (inputs, labels) in enumerate(galleryloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        gallery_features = torch.cat((gallery_features, features), dim=0)
        gallery_labels = torch.cat((gallery_labels, labels))

gallery_labels -= 2

# save features
features = {
    "qf": query_features.view(query_features.size()[0],-1),
    "ql": query_labels,
    "gf": gallery_features.view(gallery_features.size()[0],-1),
    "gl": gallery_labels
}
torch.save(features, "features.pth")
