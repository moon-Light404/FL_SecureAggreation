import os
# s = 'Hello World的法务'
# f = open('test.txt', 'a+', encoding="utf-8")
# f.write(s)
# f.close()


def f(a,b):
    return int(a) - int(b)

filepath = "F:/pythonProject/federated-learning-master/TestResult/"

path = os.listdir(filepath)
path.sort(key=lambda x: int(x))
print(path)
file_last = int(os.listdir(filepath)[-1]) + 1
print(file_last)
# new_filepath = filepath + str(file_last)
# os.mkdir(new_filepath)
# f = open(new_filepath + '/info.txt', 'a', encoding='utf-8')
# f.write("fdswafdw")