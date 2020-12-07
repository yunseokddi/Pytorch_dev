import pandas as pd
import matplotlib.pyplot as plt

from tools.plot_tool import plot, class_spec
from tools.preprocessing import *

trainpath = './data/mitbih_train.csv'
testpath = "./data/mitbih_test.csv"

x_train = pd.read_csv(trainpath, header=None, usecols=range(187))
y_train = pd.read_csv(trainpath, header=None, usecols=[187]).iloc[:, 0]

x_test = pd.read_csv(testpath, header=None, usecols=range(187))
y_test = pd.read_csv(testpath, header=None, usecols=[187]).iloc[:, 0]

# ----------check the data--------
# plot(x_train, y_train)

#----------check the target--------
# for i in range(4):
#     class_spec(x_train, i, 400, y_train)

#----------check the train target distibution--------
# y_train.value_counts().plot(kind="bar", title="y_train")

#----------check the test target distibution--------
# y_test.value_counts().plot(kind="bar", title="y_test")

#----------check the after gaussian smooting for all data avg--------
# fig = plt.figure(figsize=(8,4))
# plt.plot(x_train.iloc[1,:], label="original")
# plt.plot(gauss_wrapper(x_train.iloc[1,:]), label="smoothed")
# plt.legend()
# plt.show()

#----------check the after gaussian smooting for each class--------
# x_train_grad = gradient(x_train)
# x_test_grad = gradient(x_test)
#
# x_train_preprocessed = preprocess(pd.concat([x_train, x_train_grad, gradient(x_train_grad)], axis=1))
# x_test_preprocessed = preprocess(pd.concat([x_test, x_test_grad, gradient(x_test_grad)], axis=1))
#
# plot(x_train_preprocessed, y_train)
# plt.show()