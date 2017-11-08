import os, shutil, random


for i in range(10):
    #pass
    os.mkdir('../../data/val/'+str(i))
    os.mkdir('../../data/train/' + str(i))


all_f = '../../data/original/all/'
for fnm in os.listdir(all_f):
    if fnm.endswith('.png'):
        if random.random() > 0.75:
            shutil.copyfile(all_f + fnm, '../../data/val/' + fnm[0]+'/'+fnm)
        else:
            shutil.copyfile(all_f + fnm, '../../data/train/' + fnm[0]+'/'+fnm)
