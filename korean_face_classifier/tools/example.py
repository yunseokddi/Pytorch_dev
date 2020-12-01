import os

for root, dirs, files in os.walk('./sample_data'):
    for fname in files:
        full_fname = os.path.join(root,fname)
        target = full_fname.split('/')
        print(target[2])
        if 'jpg' in fname:
            print(fname)