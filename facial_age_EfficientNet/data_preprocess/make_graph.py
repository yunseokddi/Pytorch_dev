import os
import matplotlib.pyplot as plt
import argparse

class analysis_data:
    def __init__(self, root_dir, start_age, end_age):
        self.root_dir = root_dir
        self.start_age = start_age
        self.end_age = end_age+1
        self.x = [i for i in range(self.start_age, self.end_age)]

    def get_file_len_list(self):
        folder_len = self.end_age - self.start_age + 1
        count = 0
        file_num = []

        for i in range(folder_len):
            path = self.root_dir+'/'+str(i)

            for pack in os.walk(path):
                for f in pack[2]:
                    count += 1

                file_num.append(count)
                count = 0

        return file_num

    def get_avg_len(self, file_num):
        avg = 0

        for num in file_num:
            avg += num

        avg /= len(file_num)

        return avg

    def get_file_len(self, file_num):
        return len(file_num)

    def draw_file_len_plot(self, file_num):
        plt.plot(self.x, file_num)
        plt.xlabel('age')
        plt.ylabel('data num')
        plt.xticks([i for i in range(0, 91, 5)])

    def draw_avg_len_plt(self, avg):
        avg_y = [avg for i in range(self.start_age, self.end_age)]
        plt.plot(self.x, avg_y)

    def imshow_plot(self):
        plt.title('Facial age data')
        plt.show()

    def draw_bar(self, file_num):
        plt.title('Facial age data')
        plt.xlabel('age')
        plt.ylabel('data num')
        plt.xticks([i for i in range(0,91, 5)])
        plt.bar(self.x, file_num)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_age', type=int, help='enter start age')
    parser.add_argument('--end_age', type=int, help='enter end age')
    parser.add_argument('--root_dir', type=str, help='enter root dir')

    opt = parser.parse_args()

    analysis = analysis_data(opt.root_dir, opt.start_age, opt.end_age)
    file_num = analysis.get_file_len_list()
    avg = analysis.get_avg_len(file_num)
    analysis.draw_file_len_plot(file_num)
    analysis.draw_avg_len_plt(avg)
    analysis.imshow_plot()
    analysis.draw_bar(file_num)