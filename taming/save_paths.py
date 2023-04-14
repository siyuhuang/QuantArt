import os

rootdir = '../datasets/painter-by-numbers/test'
f = open("../datasets/painter-by-numbers-test.txt", "w")
for file in sorted(os.listdir(rootdir)):
    if file.endswith('.jpg') or file.endswith('.png'):
        f.write(os.path.join(rootdir, file)+'\n')
f.close()