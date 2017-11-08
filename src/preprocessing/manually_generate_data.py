from PIL import Image
import numpy as np
from src.utils.digit_painter import PaintWindow

SMPL_PER_CLASS = 40
ALREADY_CREATED_PER_CLASS = 40
FILENAME = 'C:/Users/eemko/OneDrive/PWr/Semestr VII/Sieci Neuronowe/Lab/Lab2/data/my_generated/%d/%d.png'

for num in range(10):
    for i in range(SMPL_PER_CLASS):
        print('Draw number %d - %d of %d' % (num, i+1, SMPL_PER_CLASS))
        ptw = PaintWindow()
        arr = ptw()
        img = Image.fromarray(np.uint8(arr)*255).convert('1')
        filename = FILENAME % (num, ALREADY_CREATED_PER_CLASS+i)
        img.save(filename, 'PNG')
