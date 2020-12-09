import os
import pandas as pd
import numpy as np


def load_ECG_dataset(path):
    csv_train_path = os.path.join(path, 'mitbih_train.csv')
    csv_test_path = os.path.join(path, 'mitbih_test.csv')
    csv_train_data = pd.read_csv(csv_train_path)
    csv_test_data = pd.read_csv(csv_test_path)

    train_x = np.array(csv_train_data.iloc[:, :187], dtype=np.float32).reshape(-1, 187, 1)
    train_y = np.array(csv_train_data.iloc[:, 187], dtype=np.int32)

    test_x = np.array(csv_test_data.iloc[:, :187], dtype=np.float32).reshape(-1, 187, 1)
    test_y = np.array(csv_test_data.iloc[:, 187], dtype=np.int32)

    Nonectopic_beat, Ventricular_ectopic_beat, Fusion_beat, Unknown_beat = [], [], [], []

    for data, label in zip(train_x, train_y):
        if label == 0:
            Nonectopic_beat.append([data, label])
        elif label == 1:
            continue
        elif label == 2:
            Ventricular_ectopic_beat.append([data, label])
        elif label == 3:
            Fusion_beat.append([data, label])
        elif label == 4:
            Unknown_beat.append([data, label])

    train_dataset_dict = {"N": Nonectopic_beat,
                          "V": Ventricular_ectopic_beat,
                          "F": Fusion_beat,
                          "Q": Unknown_beat}

    print("---Training dataset---")
    print("Nonectopic beat               :", len(Nonectopic_beat))
    print("Ventricular ectopic beat      :", len(Ventricular_ectopic_beat))
    print("Fusion beat                   :", len(Fusion_beat))
    print("Unknown beat                  :", len(Unknown_beat))
    print("----------------------")

    Nonectopic_beat, Ventricular_ectopic_beat, Fusion_beat, Unknown_beat = [], [], [], []

    for data, label in zip(test_x, test_y):
        if label == 0:
            Nonectopic_beat.append([data, label])
        elif label == 1:
            continue
        elif label == 2:
            Ventricular_ectopic_beat.append([data, label])
        elif label == 3:
            Fusion_beat.append([data, label])
        elif label == 4:
            Unknown_beat.append([data, label])

    test_dataset_dict = {"N": Nonectopic_beat,
                         "V": Ventricular_ectopic_beat, "F": Fusion_beat, "Q": Unknown_beat}

    print("---Test dataset---")
    print("Nonectopic beat               :", len(Nonectopic_beat))
    print("Ventricular ectopic beat      :", len(Ventricular_ectopic_beat))
    print("Fusion beat                   :", len(Fusion_beat))
    print("Unknown beat                  :", len(Unknown_beat))
    print("----------------------")

    return train_dataset_dict, test_dataset_dict


def split_dataset(train_dataset_dict, val_num, seed=0):
    Nonectopic = train_dataset_dict["N"]
    Ventricular = train_dataset_dict["V"]
    Fusion = train_dataset_dict["F"]
    Unknown = train_dataset_dict["Q"]

    train, validation = [], []

    np.random.seed(seed)
    np.random.shuffle(Nonectopic)
    np.random.shuffle(Ventricular)
    np.random.shuffle(Fusion)
    np.random.shuffle(Unknown)

    data_list = [Nonectopic, Ventricular, Fusion, Unknown]

    for i, dataset in enumerate(data_list):
        for data, label in dataset[:val_num]:
            validation.append([data, label])

        for data, label in dataset[val_num:]:
            train.append([data, label])

    print("trian: {}".format(len(train)))
    print("validation: {}".format(len(validation)))

    return train, validation

train_dataset_dict, test_dataset_dict = load_ECG_dataset('./data')
train_dataset, validation_dataset = split_dataset(train_dataset_dict, val_num=100, seed=0)
