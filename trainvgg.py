from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
import torch 
from models.vgg16 import Vgg16
import torchvision.models as models
from torchvision.datasets import ImageFolder
from matplotlib import pyplot as plt
import os

model = models.vgg16(pretrained=True)
       

for param in model.parameters():
    param.require_grad = False
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.Linear(4096,1024),
    nn.Linear(1024,10)
)
model.cuda()
model_path = '/data1/weiheng3_7_2021/statefarm/vggckpt2/'
early_stop_steps = 10

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


train_dataloader = DataLoader(dataset=train_set,shuffle=True, batch_size=32, num_workers=0)
test_dataloader = DataLoader(dataset=val_set,shuffle=True, batch_size=32, num_workers=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_print = []
val_acc = []
best_acc = 0
save_model_name = os.path.join(model_path,'best_model.pth')
last_model_name = os.path.join(model_path,'last_model.pth')
flag = 0
for epoch in range(100):
    print(epoch)
    model.train()
    epoch_loss = 0
    # countbreak = 0
    for (data_x, label) in train_dataloader:
        optimizer.zero_grad()
        #pdb.set_trace()
        input = data_x
        label = label
        input = input.cuda()
        label = label.cuda()
    
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_print.append(epoch_loss)
        # if countbreak >=3:
        #     break
        # countbreak+=1

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
            output = model(input)
            pred = torch.argmax(output,1)
            correct = torch.sum(pred==label)
            corrects += correct.item()
            total += input.shape[0]
          
    print(corrects/total)
    val_acc.append(corrects/total)
    if best_acc <= corrects/total:
        torch.save(model.state_dict(),save_model_name)
        flag = 0
    else:
        flag += 1
        if flag >= 10:
            torch.save(model.state_dict(),last_model_name)
            break
torch.save(model.state_dict(),last_model_name)
x_list= []
for i in range(len(loss_print)):
    x_list.append(i)
plt.plot(x_list,loss_print)
plt.savefig('./resloss.png')
plt.clf()
x_list= []
for i in range(len(val_acc)):
    x_list.append(i)
plt.plot(x_list,val_acc)
plt.savefig('./resacc.png')
    
# if __name__ == '__main__':

#     '''加载数据'''
#     train_data_path = cfg.TRAIN.train_data_path

#     train_data = train_data_loader.DriverDataset(train_data_path, train=True)
#     train_dataloader = DataLoader(dataset=train_data,shuffle=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)

#     vali_data = train_data_loader.DriverDataset(train_data_path, train=False)
#     vali_dataloader = DataLoader(dataset=vali_data, shuffle=False, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)
    
#     print(model)
    
#     '''训练'''
#     loss_print = []
#     j = 0

#     f = open('./V2_m_loss.txt','w')
#     for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCHS):
#         st = time.time()
#         for (data_x, label) in train_dataloader:
#             j += 1
#             optimizer.zero_grad()
#             #pdb.set_trace()
#             input = data_x
#             label = label
#             if cfg.TRAIN.use_gpu:
#                 input = input.cuda()
#                 label = label.cuda()
        
#             output = model(input)
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
#             loss_print.append(loss)
            
#             '''print loss'''
#             if j % cfg.TRAIN.frequency_print == 0:
#                 loss_mean = t.mean(t.Tensor(loss_print))
#                 loss_print = []
#                 print('第 %d epoch, step : %d' % (epoch, j), 'train_loss: %f'%loss_mean)
#                 f.write(str(j)+','+'%f'%loss_mean+'\n')

#         print("epoch time : %f s" % (time.time()-st))
#         '''可视化模型在验证集上的准确率'''
#         acc_vali = val(model, vali_dataloader)
#         print('第 %d epoch, acc_vali : %f' % (epoch, acc_vali))

#         '''每epoch,保存已经训练的模型'''
#         trainedmodel_path = './trained_models/V2_m/'
#         if not os.path.isdir(trainedmodel_path):
#             os.makedirs(trainedmodel_path)
#         t.save(model, trainedmodel_path + '%d'%epoch + '_' + '%f'%acc_vali + '.pkl')
#     f.close()

        
