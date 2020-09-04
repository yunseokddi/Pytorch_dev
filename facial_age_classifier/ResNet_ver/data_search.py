import os
import matplotlib.pyplot as plt

root_data_path = './data/'
count = 0
file_num = []
avg = 0

for i in range(0, 61, 10):
    if i == 50:
        continue

    path = root_data_path + str(i)

    for pack in os.walk(path):
        for f in pack[2]:
            count += 1

        file_num.append(count)
        count = 0

for num in file_num:
    avg += num

avg /= len(file_num)

x = [i for i in range(0, 61, 10)]
x.remove(50)
avg_y = [avg for i in range(0, 51, 10)]


plt.plot(x, file_num, label='data num')
plt.plot(x, avg_y, label='avg')

plt.xlabel('age')
plt.ylabel('data num')

plt.title('Facial age data')
plt.show()
