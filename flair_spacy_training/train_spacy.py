import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
os.environ["CUDA_VISIBLE_DEVICES"]="3";
 

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from helper import convert_dataturks_to_spacy, train_spacy,Stanford_to_dataturks_format

sns.set()

spacy.util.use_gpu(0)
stanford_path = "data/BC5CDR-chem/train.tsv"
output_path = "data/BC5CDR-chem/spacy_train.json"

Stanford_to_dataturks_format(stanford_path,output_path,"UNK")
print(spacy.prefer_gpu())
DATA_PATH = output_path
TRAIN_DATA = convert_dataturks_to_spacy(DATA_PATH)
nlp, history = train_spacy(TRAIN_DATA)
path_model = './BC5CDR-chem_model'
nlp.to_disk(path_model)
import numpy
a = numpy.asarray(history)
numpy.savetxt("loss-BC5CDR-chem.csv", a, delimiter=",")



stanford_path = "data/JNLPBA/train.tsv"
output_path = "data/JNLPBA/spacy_train.json"

Stanford_to_dataturks_format(stanford_path,output_path,"UNK")
print(spacy.prefer_gpu())
DATA_PATH = output_path
TRAIN_DATA = convert_dataturks_to_spacy(DATA_PATH)
nlp, history = train_spacy(TRAIN_DATA)
path_model = './JNLPBA_model'
nlp.to_disk(path_model)
import numpy
a = numpy.asarray(history)
numpy.savetxt("lossJNLPBA.csv", a, delimiter=",")

plt.subplots(figsize=(12, 6))
plt.plot(history)
plt.title('Loss History')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()