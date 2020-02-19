import numpy as np
import pickle
from scipy.fftpack import fft
import os
import pandas

if not os.path.exists("process"):
    os.makedirs("process")
path = 'E:\\数据\\Mechanical-datasets\\gearbox\\gearset\\'
pathdir = os.listdir(path)
rowdata1 = []
rowdata2 = []
rowdata3 = []
folds = 5
i = 0
for filename in pathdir[1:]:
    name = path + filename
    file = pandas.read_csv(name, sep='\t', header=16)
    file = file.to_numpy()
    rowdata1.append(file[:, 1])
    rowdata2.append(file[:, 2])
    rowdata3.append(file[:, 3])
    i += 1
rowdata1 = np.array(rowdata1)
rowdata2 = np.array(rowdata2)
rowdata3 = np.array(rowdata3)

channel_num = 1
for channel in [rowdata1, rowdata2, rowdata3]:
    data = channel[:, :2048 * 500].reshape(-1, 500, 2048)
    dat = list(np.ones(10))
    alldata = []

    for k in range(10):
        dat[k] = [(k, abs(fft(x - np.mean(x)))[:1024]) for x in data[k]]
        alldata+=dat[k]
    output = open('process\\%ss.pkl' % channel_num, 'wb')
    pickle.dump(alldata, output)
    output.close()

    for fold in range(folds):
        folding = []
        for k in range(10):
            temp = dat[k]
            folding += temp[(0 + 100 * fold):(100 + 100 * fold)]
        output = open('process\\%ss_fold_%d.pkl' % (channel_num, fold), 'wb')
        pickle.dump(folding, output)
        output.close()
    channel_num += 1

