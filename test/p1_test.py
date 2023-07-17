import sys
import json
import os
import os.path
import time

from Nets import CNNMnist
from Server import Server

time1=time.time()

# def MergeTxt(filepath):
#     for parent, dirnames, filenames in os.walk(filepath):
#         for filepath in filenames:
#             txtPath = os.path.join(parent, filepath) # txtpath就是所有文件夹的路径
#             f_name = os.path.splitext(filepath)[0]
#             pyPath = os.path.join(parent, f_name)
#             with open(txtPath,"rb") as fp:
#                 content = json.load(fp)
#                 #print(content)
#                 with open(pyPath + ".py","w",encoding= 'utf-8') as fp:
#                     for item in content["cells"]:
#                         for i in item['source']:
#                             fp.write(i.rstrip() +"\n")
#                     fp.close()
#                     print(txtPath + '转换完成')
#
# print ("finished")
# if __name__ == '__main__':
#     filepath="F:\\bishe" #注意：应先删除文件大小为0的文件。
#     MergeTxt(filepath)
#     time2 = time.time()
#     print ('总共耗时：' + str(time2 - time1) + 's')

server = CNNMnist()

print(server.state_dict())