import os
import torch
from net.utils import EMRCSDataset
import torch.utils.data.dataloader as DataLoader

rcsdir = r'/mnt/truenas_jiangxiaotian/allplanes/mie/b943_mie_train'
dataset = EMRCSDataset(rcsdir)
dataloader = DataLoader.DataLoader(dataset, batch_size=12, shuffle=False, num_workers=16, pin_memory=True)

for i, data in enumerate(dataloader):
    rcs, label = data # label = [plane,theta,phi,freq]
    ave_rcs = rcs.mean(dim=0, keepdim=True)  # Average RCS across the batch
    
    # print(f'Batch {i+1}:')
    # print(f'RCS shape: {rcs.shape}, Label shape: {label.shape}')
    # if i == 2:  # Limit to 3 batches for demonstration
    #     break
