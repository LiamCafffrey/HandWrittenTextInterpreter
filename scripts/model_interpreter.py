import pandas as pd

mapp = pd.read_csv("../raw_data/emnist-byclass-mapping.txt", delimiter = ' ',index_col=0, header=None, squeeze=True)

def interpreter(prediction):
  return chr(mapp[prediction])
