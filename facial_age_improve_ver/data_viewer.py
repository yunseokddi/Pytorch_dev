import os
import matplotlib.pyplot as plt

root_data_path = 'data/past_data/face_age/'
count = 0
file_num = []
avg = 0
sum = 0

for i in range(10, 51, 10):
    path = root_data_path + str(i)

    for pack in os.walk(path):
        for f in pack[2]:
            count += 1
            sum += 1

        file_num.append(count)
        count = 0

for num in file_num:
    avg += num

avg /= len(file_num)

print('folder list: {}'.format(file_num))
print('sum: {}'.format(sum))
print('avg: {}'.format(round(avg)))

x = [i for i in range(10, 51, 10)]
avg_y = [avg for i in range(10, 51, 10)]

plt.plot(x, file_num, label='data num')

plt.xlabel('age')
plt.ylabel('data num')

plt.title('Facial age data')
plt.show()
