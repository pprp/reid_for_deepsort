import torch
import numpy as np

features = torch.load("features.pth")

qf = features["qf"]
ql = features["ql"]
gf = features["gf"]
gl = features["gl"]

scores = qf.mm(gf.t())
# scores = np.dot(gf, qf)
# print(ql[:3])

res = scores.topk(20, dim=1)[1][:,0]

print(gl[res])
print(gf[res])

top1correct = gl[res].eq(ql).sum().item()

print("Acc top1:{:.3f}".format(top1correct/ql.size(0)))


