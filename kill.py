import os

lines = os.popen('ps aux | grep train_all').readlines()
print(lines)
for line in lines:
    line = line.split()
    p = line[1]
    print(f'killing {p}')
    os.system(f'kill -9 {p}')
