import pandas as pd
import sys
import os
from PIL import Image
from tqdm import tqdm

label = sys.argv[1]
lis = os.listdir("../data/csm/{}".format(label))
p_bar = tqdm(total = len(pair_file))

for each in lis:
    p_bar.update(1)
    val = pd.read_csv(each, header=None).values
    img = Image.fromarray(np.uint8(val * 255) , 'L')
    img.save("training_dataset/{}/{}{}".format(label,each.split(".")[0], '.jpg'))

p_bar.close()

