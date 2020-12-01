import os
import shutil


def mv_files(dir):
    count = 0
    for root, dirs, files in os.walk(dir):
        for fname in files:
            full_fname = os.path.join(root, fname)
            if 'jpg' in fname:
                target = full_fname.split('/')
                idx = full_fname.find(target[-1])
                rename_dir =  os.path.join(full_fname[:idx],target[3] + '_' + str(count) + '.jpg')
                moved_dir = os.path.join(dir,target[3],target[3] + '_' + str(count) + '.jpg')
                os.rename(full_fname, rename_dir)
                shutil.move(rename_dir,moved_dir)
                count += 1
#
# def rm_empty_folder(dir):
#     for root, dirs, files in os.walk(dir):
#         rm_dir = root.split('/')
#         # print(len(rm_dir))
#         # print(rm_dir)
#         if len(rm_dir) > 6:
#             rm_dir = os.path.join(rm_dir[0],rm_dir[1],rm_dir[2],rm_dir[3])
#             print(rm_dir)


if __name__ == '__main__':
    mv_files('./sample_data/image/')
    # rm_empty_folder('./sample_data/image/')