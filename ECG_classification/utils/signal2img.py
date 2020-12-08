import pandas as pd
import numpy as np
import biosppy as bp
import os
import cv2
import matplotlib.pyplot as plt

# change directory path
path = './F/'
directory = '../data/2_class_img_data/F2'


def main(path, directory, count):
    def segmentation(path, count):
        csv = pd.read_csv(path, engine='python')
        csv_data = csv['Sample Value']
        data = np.array(csv_data)
        signals = []
        peaks = bp.signals.ecg.christov_segmenter(signal=data, sampling_rate=200)[0]
        for i in (peaks[1:-1]):
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1] - i)
            x = peaks[count - 1] + diff1 // 2
            y = peaks[count + 1] - diff2 // 2
            signal = data[x:y]
            signals.append(signal)
            count += 1
        return signals

    def signal_to_img(array, directory):
        for count, i in enumerate(array):
            fig = plt.figure(frameon=False)
            plt.plot(i)
            plt.xticks([]), plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            filename = directory + '/' + str(count) + '.png'
            fig.savefig(filename)
            print(filename)
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(filename, im_gray)
            plt.close(fig)
        return directory

    array = segmentation(path, count)
    directory = signal_to_img(array, directory)
    return directory

if __name__ == '__main__':
    count = 1
    # for root, dirs, files in os.walk(path):
    #     for fname in files:
    #         print(path+fname)
    #         directory = main(path + fname, directory, count)

    directory = main('./F/Ftype2.csv', directory, count)