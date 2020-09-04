import shutil
import os

for i in range(10, 56, 5):
    sourse = './data/' + str(i)
    dest = './data/' + str(i)
    for i in range(i + 1, i + 5):
        files = os.listdir(sourse + '/' + str(i) + '/')
        for f in files:
            print(sourse + '/' + str(i) + '/' + f)
            shutil.move(sourse + '/' + str(i) + '/' + f, dest)
