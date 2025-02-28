import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from  sentance_preprocess import read_language

SOS_token = 0
EOS_token = 1

lang_input, lang_output, pairs = read_language('ENG', 'KOR', reverse=False, verbose=False)
for idx in range(10):
    print(random.choice(pairs))

