import os

file_list = os.listdir('./data/train/labels')
src_path = './data/train/labels/'

class_dict = {'Car': 0,
              'Van': 1,
              'Truck': 2
              }

for file in file_list:
    f = open(src_path + file, 'r', encoding='UTF8')
    Read = f.read()

    Iteration_find = ['Dont0e']
    Iteration_Replace = ['3']
    Iteration_Merge = [(Iteration_find[n], Iteration_Replace[n]) for n in range(0,len(Iteration_find))]

    for (i,j) in Iteration_Merge:
        Read = Read.replace('{}'.format(i), '{}'.format(j))

    f = open(src_path+file, 'w', encoding='UTF8')
    f.write(Read)
    f.close()

