import os

path = '../data/2_class_img_data/F2/'

count = 2267

for root, dirs, files in os.walk(path):
    for fname in files:
        print(fname)
        os.rename(path + fname, path + str(count) + '.png')
        count += 1
