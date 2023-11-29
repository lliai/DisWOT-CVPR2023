import argparse
import logging
import os
import random
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.imagenet import get_imagenet_dataloader
from models.candidates.mutable import MasterNet

masternet = MasterNet(
    num_classes=1000,
    opt=None,
    argv=None,
    no_create=False,
    plainnet_struct=
    'SuperConvK3BNRELU(3,32,2,1)SuperResK1K3K1(32,32,2,16,2)SuperResK1K3K1(32,32,2,16,2)SuperResK1K3K1(32,32,1,16,2)SuperResK1K3K1(32,32,2,16,2)SuperConvK1BNRELU(32,32,1,1)'
    # 'SuperConvK3BNRELU(3,8,2,1)SuperResK1K7K1(8,64,2,32,1)SuperResK1K3K1(64,152,2,128,1)SuperResK1K5K1(152,640,2,192,4)SuperResK1K5K1(640,640,1,192,2)SuperResK1K5K1(640,1536,2,384,4)SuperResK1K5K1(1536,816,1,384,3)SuperResK1K5K1(816,816,1,384,3)SuperConvK1BNRELU(816,5304,1,1)'
)

masternet = masternet.cuda()

train_loader, val_loader = get_imagenet_dataloader(dataset='imagenet',
                                                   batch_size=32,
                                                   num_workers=16,
                                                   is_instance=False)

print(masternet)

for image, label in train_loader:
    image = image.cuda()
    masternet(image)
