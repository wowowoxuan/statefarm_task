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
model = weightcomb().cuda()
#mobilenetv2ckpt2
model_path = '/data1/weiheng3_7_2021/statefarm/fc_ckpt'
mob_path = '/data1/weiheng3_7_2021/statefarm/mobilenetv2ckpt3/best_model.pth'
res_path = '/data1/weiheng3_7_2021/statefarm/res2/best_model.pth'
early_stop_steps = 10

# test_path = '/data1/weiheng3_7_2021/statefarm/imgs/test/'
# test_path = '/data1/weiheng3_7_2021/statefarm/testfolder/c0/'

train_path = '/data1/weiheng3_7_2021/statefarm/imgs/train/'
val_path = '/data1/weiheng3_7_2021/statefarm/imgs/val/'



val_trans = T.Compose([T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
train_trans = T.Compose([T.RandomHorizontalFlip(),
                                         T.RandomResizedCrop(224),
                                         T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                         T.RandomRotation(degrees=60, resample=False, expand=False),
                                         T.ToTensor(),
                                         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
train_set = ImageFolder(root = train_path,transform = train_trans)
val_set = ImageFolder(root = val_path,transform = val_trans)


train_dataloader = DataLoader(dataset=train_set,shuffle=True, batch_size=128, num_workers=0)
test_dataloader = DataLoader(dataset=val_set,shuffle=True, batch_size=128, num_workers=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss_print = []
val_acc = []
best_acc = 0
save_model_name = os.path.join(model_path,'best_model.pth')
last_model_name = os.path.join(model_path,'last_model.pth')
flag = 0

mob.load_state_dict(torch.load(mob_path))
mob.eval()
res.load_state_dict(torch.load(res_path))
res.eval()

for epoch in range(100):
    print(epoch)
    model.train()
    epoch_loss = 0
    for (data_x, label) in train_dataloader:
        optimizer.zero_grad()
        input = data_x
        label = label
        input = input.cuda()
        label = label.cuda()
        with torch.no_grad():
            moboutput = mob(input)
            resoutput = res(input)
            next_in = torch.cat((moboutput,resoutput),1)

        
    
        output = model(next_in)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_print.append(epoch_loss)

    with torch.no_grad():
        total = 0
        corrects = 0
        model.eval()
        for (data_x, label) in test_dataloader:

        #pdb.set_trace()
            input = data_x
            label = label
            input = input.cuda()
            label = label.cuda()
            moboutput = mob(input)
            resoutput = res(input)
            next_in = torch.cat((moboutput,resoutput),1)

            output = model(next_in)
            pred = torch.argmax(output,1)
            correct = torch.sum(pred==label)
            corrects += correct.item()
            total += input.shape[0]

          
    print(corrects/total)        
    val_acc.append(corrects/total)
    if best_acc <= corrects/total:
        print('model saved')
        best_acc = corrects/total
        torch.save(model.state_dict(),save_model_name)
        flag = 0
    else:
        flag += 1
        print(flag)
        if flag >= 10:
            torch.save(model.state_dict(),last_model_name)
            break

torch.save(model.state_dict(),last_model_name)
x_list= []
for i in range(len(loss_print)):
    x_list.append(i)
plt.plot(x_list,loss_print)
plt.savefig('./ensambleloss.png')
plt.clf()
x_list= []
for i in range(len(val_acc)):
    x_list.append(i)
plt.plot(x_list,val_acc)
plt.savefig('./ensambleacc.png')









# mob.load_state_dict(torch.load(mob_path))
# mob.eval()
# res.load_state_dict(torch.load(res_path))
# res.eval()
# m = nn.Softmax()
# with torch.no_grad():
#     for (data_x, label) in test_dataloader:
#         input = data_x
#         print(label)
#         input = input.cuda()

#         moboutput = mob(input)
#         resoutput = res(input)
#         output = m(0.2*moboutput + 0.8*resoutput)
#         out_np = output.cpu().numpy()
#         for i in range(len(label)):
#             pre = out_np[i].tolist()
#             newlist = []
#             lb = label[i].replace(test_path,'')
    
#             newlist.append(lb)
#             for item in pre:
#                 newlist.append(str(item))
#             with open(path,'a') as f:
#                 csv_write = csv.writer(f)
#                 csv_write.writerow(newlist)


        

        
