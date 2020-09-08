import os
import shutil
import random
import os.path

src_dir = '../data/class_15_data/train/70'
target_dir = '../data/class_15_data/val/70/'
src_files = (os.listdir(src_dir))

def valid_path(dir_path, filename):
       full_path = os.path.join(dir_path, filename)
       return os.path.isfile(full_path)


files = [os.path.join(src_dir, f) for f in src_files if valid_path(src_dir, f)]
choices = random.sample(files, 370)
for files in choices:
       shutil.move(files, target_dir)
print ('Finished!')