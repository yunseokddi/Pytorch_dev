import random
import os

num = 1500

for file in range(0,6001):
    num = str(file).zfill(6)
    print('./data/train/images/{}.png'.format(num))