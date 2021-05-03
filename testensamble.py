from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
import torch 
from models.mobilenetv2 import Mobilenet_v2
from models.weightcomb import weightcomb
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import os
from testset import DriverDataset
import csv
import torchvision


res = torchvision.models.resnet34(pretrained=True)
res.fc = nn.Linear(512, 10)#512为resnet34倒数第二层神经元的个数
res.cuda()

mob = Mobilenet_v2().cuda()
weightcomb = weightcomb().cuda()
#mobilenetv2ckpt2
mob_path = '/data1/weiheng3_7_2021/statefarm/mobilenetv2ckpt3/best_model.pth'
model_path = '/data1/weiheng3_7_2021/statefarm/fc_ckpt/best_model.pth'
res_path = '/data1/weiheng3_7_2021/statefarm/res2/best_model.pth'
early_stop_steps = 10


# test_path = '/data1/weiheng3_7_2021/statefarm/imgs/test/'
# test_path = '/data1/weiheng3_7_2021/statefarm/testfolder/c0/'

train_path = '/data1/weiheng3_7_2021/statefarm/imgs/train/'
val_path = '/data1/weiheng3_7_2021/statefarm/imgs/val/'
test_path = '/data1/weiheng3_7_2021/statefarm/imgs/test/'

path = "./ensamblenn.csv"
with open(path,'w') as f:
    csv_write = csv.writer(f)
    csv_head = ["img","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]
    csv_write.writerow(csv_head)

test_set = DriverDataset(test_path)


test_dataloader = DataLoader(dataset=test_set,shuffle=False, batch_size=400, num_workers=0)

mob.load_state_dict(torch.load(mob_path))
mob.eval()
res.load_state_dict(torch.load(res_path))
res.eval()
weightcomb.load_state_dict(torch.load(model_path))
m = nn.Softmax()
with torch.no_grad():
    for (data_x, label) in test_dataloader:
        input = data_x
        print(label)
        input = input.cuda()

        moboutput = mob(input)
        resoutput = res(input)
        next_in = torch.cat((moboutput,resoutput),1)
        output = weightcomb(next_in)
        output = m(output)
        out_np = output.cpu().numpy()
        for i in range(len(label)):
            pre = out_np[i].tolist()
            newlist = []
            lb = label[i].replace(test_path,'')
    
            newlist.append(lb)
            for item in pre:
                newlist.append(str(item))
            with open(path,'a') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(newlist)


        

        
