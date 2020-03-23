# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-03-19T12:27:44+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-03-23T13:29:47+08:00

import sys

file = open(sys.argv[1])
file = file.readlines()
best = 0
epoch_best = 0
for line in file:
    if 'Eval epoch' in line:
        epoch_number = line.split('epoch:')[1]
    elif 'Top1' in line:
        accuracy = float(line.split('Top1:')[1].split('%')[0])
#         print(int(epoch_number) , ' ',accuracy)
        if best<accuracy:
            epoch_best = int(epoch_number)
            best =  accuracy
    else:
        continue
print(best)
print(epoch_best)
