import os
import shutil


for i in range(10):
    for fnm in os.listdir():
        if fnm.endswith('.png') and fnm[0]== str(i):
            shutil.copyfile(fnm, str(i)+'/'+fnm)