import os
import shutil

def change_folder_name(ROOT_PATH): #extract left 'zeors'
    for name in os.listdir(ROOT_PATH):
        new_name = name.lstrip('0')
        os.rename(ROOT_PATH+name, ROOT_PATH+new_name)

def folder_merge(ROOT_PATH):
    for i in range(10, 51, 10):
        sourse = ROOT_PATH + str(i)
        dest = ROOT_PATH + str(i)
        for i in range(i + 1, i + 10):
            files = os.listdir(sourse + '/' + str(i) + '/')
            for f in files:
                print(sourse + '/' + str(i) + '/' + f)
                shutil.move(sourse + '/' + str(i) + '/' + f, dest)

def sub_folder_delete(ROOT_PATH):
    for i in range(20, 51, 10):
        sourse = ROOT_PATH + str(i)
        for i in range(i + 1, i + 10):
            print(sourse + '/' + str(i))
            os.rmdir(sourse + '/' + str(i))

if __name__ == '__main__':
    # change_folder_name('./data/face_age/')
    # folder_merge('./data/face_age/')
    sub_folder_delete('./data/face_age/')