import cv2
import numpy as np
import pandas as pd

from DataLoader import k_face_dataset

k_face_loader = k_face_dataset(root_dir='./sample_data', csv_file='./sample_data/KFace_data_information_Folder1_400.xlsx')

print(len(k_face_loader))

print(k_face_loader[2])