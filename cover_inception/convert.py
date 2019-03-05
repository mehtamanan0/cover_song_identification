import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from tqdm import tqdm

label = sys.argv[1]
direc = "../data/csm/{}".format(label)
lis = os.listdir(direc)
p_bar = tqdm(total = len(lis))

for each in lis:
    p_bar.update(1)
    val = pd.read_csv(os.path.join(direc, each), header=None).values
    img = Image.fromarray(np.uint8(val * 255) , 'L')
    img.save("training_dataset/{}/{}{}".format(label,each.split(".")[0], '.jpg'))

p_bar.close()

