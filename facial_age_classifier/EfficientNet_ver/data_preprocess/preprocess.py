import os
import shutil
import argparse


class file_preprocess:
    def __init__(self, src_folder, dst_folder):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.file_list = os.listdir(src_folder)

    def move_file_by_split_age(self):
        for file in self.file_list:
            shutil.move(self.src_folder + '/' + str(file), self.dst_folder + '/' + file.split('_')[0] + '/' + str(file))

        print('File move finish for {} files'.format(len(self.file_list)))
        shutil.rmtree(self.src_folder)

    def create_folder(self):
        for file in self.file_list:
            if not (os.path.isdir(self.dst_folder+ file.split('_')[0])):
                os.makedirs(os.path.join(self.dst_folder +file.split('_')[0]))

        print('Age file create finish')

    def change_names(self):
        for filename in self.file_list:
            os.rename(self.src_folder + '/' + filename, self.dst_folder + '/' + filename.lstrip('0'))

        print('facial age file change finish')

    def facial_age_data_move(self):
        for i in range(1, 91):
            path = self.src_folder + '/' + str(i)

            for pack in os.walk(path):
                for f in pack[2]:
                    shutil.move(path+'/'+f, self.dst_folder+'/'+str(i)+'/'+f)

        print('facial age files move finish')
        shutil.rmtree(self.src_folder)

    def merge_folder(self):
        for i in range(0,90,5):
            for j in range(i+1, i+5):
                sub_path = self.src_folder+'/'+str(j)

                for files in os.listdir(sub_path):
                    shutil.move(self.src_folder+'/'+str(j)+'/'+ files, self.src_folder+'/'+str(i)+'/'+files)

        print('merge finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help='enter your src path')
    parser.add_argument('--dst_path', type=str, help='enter your dst path')
    parser.add_argument('--create_folder', default=False, action='store_true', help='create folder by age')
    parser.add_argument('--move_files', default=False, action='store_true', help='move files')
    parser.add_argument('--change_names', default=False, action='store_true', help='change the facial age file')
    parser.add_argument('--facial_age_data_move', default=False, action='store_true', help='move facial age dataset')
    parser.add_argument('--merge_files', default=False, action='store_true', help='merge files')
    opt = parser.parse_args()

    src_path = opt.src_path
    dst_path = opt.dst_path

    preprocess = file_preprocess(src_path, dst_path)

    if opt.create_folder is True:
        preprocess.create_folder()

    if opt.move_files is True:
        preprocess.move_file_by_split_age()

    if opt.change_names is True:
        preprocess.change_names()

    if opt.facial_age_data_move is True:
        preprocess.facial_age_data_move()

    if opt.merge_files is True:
        preprocess.merge_folder()