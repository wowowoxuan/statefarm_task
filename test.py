# import csv
# path = "./aa.csv"
# # with open(path,'w') as f:
# #     csv_write = csv.writer(f)
# #     csv_head = ["img","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"]
# #     csv_write.writerow(csv_head)

# with open(path,'a') as f:
#         csv_write = csv.writer(f)
#         data_row = ["1","2","3","1","2","3","1","2","3","1"]
    
#         csv_write.writerow(data_row)
import torch
x1 = torch.randn(2,3)
x2 = torch.randn(2,3)
print(x1)
print(x2)
x = torch.cat((x1,x2),1)
print(x)