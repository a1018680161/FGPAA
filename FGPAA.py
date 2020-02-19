from __future__ import print_function
import torch.utils.data
from torch import optim
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import time
import random
from torch.autograd.gradcheck import zero_gradients
import scipy.stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from logger import setlogger
import logging
import os
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ----------------------------------------about CUDA
use_cuda = torch.cuda.is_available()
device = torch.cuda.current_device()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
FloatTensor = torch.cuda.FloatTensor
IntTensor = torch.cuda.IntTensor
LongTensor = torch.cuda.LongTensor

# ----------------------------------------hyper parameter
zd_merge = False
title_size = 16
axis_title_size = 14
ticks_size = 18
power = 2.0
train_epoch = 80
Reconloss = 'mse'
norm = True
datatpye = '1s'
mode = 0
folding_id,  total_classes, folds = 0, 10, 5
batch_size = 128
zsize = 16
mnist_train = []
mnist_valid = []
random.seed(0)
inputsize = 32
if mode == 0:
    inliner_classes = [2,3]
elif mode == 1:
    inliner_classes = [3]
# ----------------------------------------about path
path = 'result\\FGPAA\\' + 'norm_' + str(norm) + '_' + datatpye + '_' + Reconloss + '\\' + 'mode' +str(
    mode) + '_f' + str(folding_id) + '_ep' + str(train_epoch)
if not os.path.exists("%s" % path):
    os.makedirs("%s" % path)
setlogger(os.path.join("%s" % path, 'train.log'))
f = open("%s\\train.log" % path, 'w')
f.truncate()
f.close()
logging.info('Running on %s', torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size])
    return Variable(x)


def extract_batch_(data, it, batch_size):
    x = data[it * batch_size:(it + 1) * batch_size]
    return x


def compute_jacobian(inputs, output):
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


def GetF1(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2.0 * precision * recall / (precision + recall)


def normto01(my_matrix):
    scaler = MinMaxScaler()
    scaler.fit(my_matrix.T)
    x = scaler.transform(my_matrix.T)
    return x.T


for i in range(folds):
    if i != folding_id:
        with open('process\\%s_fold_%d.pkl' % (datatpye, i), 'rb') as pkl:
            fold = pickle.load(pkl)
        if len(mnist_valid) == 0:
            mnist_valid = fold
        else:
            mnist_train += fold

outlier_classes = []
if mode == 0:
    for i in range(total_classes):
        if i not in inliner_classes:
            outlier_classes.append(i)
elif mode == 1:
    for i in range(1,total_classes,2):
        if i not in inliner_classes:
            outlier_classes.append(i)
print(outlier_classes)

# ---------------------------------------keep only train classes
mnist_train = [x for x in mnist_train if x[0] in inliner_classes]
random.shuffle(mnist_train)


def list_of_pairs_to_numpy(l):
    return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)


def r_pdf(x, bins, count):
    if x < bins[0]:
        return max(count[0], 1e-308)
    if x >= bins[-1]:
        return max(count[-1], 1e-308)
    id = np.digitize(x, bins) - 1
    return max(count[id], 1e-308)


logging.info('Train set size:%d'%len(mnist_train))
mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
if norm:
    mnist_train_x = mnist_train_x.reshape(-1, inputsize * inputsize)
    mnist_train_x = normto01(mnist_train_x)
mnist_train_x = mnist_train_x.reshape(-1, inputsize, inputsize)
ave_train = np.array(mnist_train_x.reshape(-1,inputsize*inputsize)).mean(axis=0)
if norm:
    G = Generator_n(zsize)
else:
    G = Generator(zsize)
setup(G)
D = Discriminator()
setup(D)
E = Encoder(zsize)
setup(E)
if zd_merge:
    ZD = ZDiscriminator_mergebatch(zsize, batch_size).to(device)
else:
    ZD = ZDiscriminator(zsize, batch_size).to(device)
setup(ZD)
if norm:
    G.weight_init(mean=0, std=0.02)
    D.weight_init(mean=0, std=0.02)
    E.weight_init(mean=0, std=0.02)
    ZD.weight_init(mean=0, std=0.02)
lrG = 0.002
lrD = 0.002
lrE = 0.002
lrGE = 0.002
lrZD = 0.002
G_optimizer = optim.Adam(G.parameters(), lr=lrG, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lrD, betas=(0.5, 0.999))
E_optimizer = optim.Adam(E.parameters(), lr=lrE, betas=(0.5, 0.999))
GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lrGE, betas=(0.5, 0.999))
ZD_optimizer = optim.Adam(ZD.parameters(), lr=lrZD, betas=(0.5, 0.999))
BCE_loss = nn.BCELoss()
MSE_loss = nn.MSELoss()
y_real_ = torch.ones(batch_size)
y_fake_ = torch.zeros(batch_size)

y_real_z = torch.ones(1 if zd_merge else batch_size)
y_fake_z = torch.zeros(1 if zd_merge else batch_size)


def train_step(step=10):
    for epoch in range(step):
        G.train()
        D.train()
        E.train()
        ZD.train()
        Gtrain_loss = 0
        Dtrain_loss = 0
        Etrain_loss = 0
        GEtrain_loss = 0
        ZDtrain_loss = 0
        epoch_start_time = time.time()

        def shuffle(X):
            np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)
        shuffle(mnist_train_x)
        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            E_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            logging.info("learning rate change!")
        for it in range(len(mnist_train_x) // batch_size):
            x = extract_batch(mnist_train_x, it, batch_size).view(-1, 1, inputsize, inputsize)
            # D
            D.zero_grad()
            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)
            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)
            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()
            Dtrain_loss += D_train_loss.item()

            # G
            G.zero_grad()
            z = torch.randn((batch_size, zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)
            x_fake = G(z)
            D_result = D(x_fake).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()
            Gtrain_loss += G_train_loss.item()

            # ZD
            ZD.zero_grad()
            z = torch.randn((batch_size, zsize)).view(-1, zsize)
            z = Variable(z)
            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)
            z = E(x).squeeze().detach()
            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)
            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()
            ZD_optimizer.step()
            ZDtrain_loss += ZD_train_loss.item()

            # GE
            E.zero_grad()
            G.zero_grad()
            z = E(x)
            x_d = G(z)
            ZD_result = ZD(z.squeeze()).squeeze()
            E_loss = BCE_loss(ZD_result, y_real_z) * 2.0

            if Reconloss == 'mse':
                Recon_loss = MSE_loss(x_d, x)
            elif Reconloss == 'bce':
                Recon_loss = F.binary_cross_entropy(x_d, x)  # Recon_loss = F.binary_cross_entropy(x_d, x)
            (Recon_loss + E_loss).backward()
            GE_optimizer.step()
            GEtrain_loss += Recon_loss.item()
            Etrain_loss += E_loss.item()

        Gtrain_loss /= (len(mnist_train_x) // batch_size)
        Dtrain_loss /= (len(mnist_train_x) // batch_size)
        ZDtrain_loss /= (len(mnist_train_x) // batch_size)
        GEtrain_loss /= (len(mnist_train_x) // batch_size)
        Etrain_loss /= (len(mnist_train_x) // batch_size)
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        logging.info('[%d/%d] - ptime: %.2f, Gloss: %.3f, Dloss: %.3f, ZDloss: %.3f, GEloss: %.3f, Eloss: %.3f' % (
            (epoch + 1), train_epoch, per_epoch_ptime, Gtrain_loss, Dtrain_loss, ZDtrain_loss, GEtrain_loss,
            Etrain_loss))
    logging.info("Training finish!... save training results")

    # pdf of z and noise
    G.eval()
    E.eval()
    if True:
        zlist = []
        rlist = []
        for it in range(len(mnist_train_x) // batch_size):
            x = Variable(extract_batch(mnist_train_x, it, batch_size).view(-1, inputsize, inputsize).data,
                         requires_grad=True)
            z = E(x.view(-1, 1, inputsize, inputsize))
            recon_batch = G(z)
            z = z.squeeze()
            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            for i in range(batch_size):
                werror = ave_train * np.array(recon_batch[i].flatten() - x[i].flatten())
                distance = np.sum(np.power(werror, power))
                rlist.append(distance)
            zlist.append(z)
        data = {}
        data['rlist'] = rlist
        data['zlist'] = zlist
        with open('process\\rz.pkl', 'wb') as pkl:
            pickle.dump(data, pkl)
    with open('process\\rz.pkl', 'rb') as pkl:
        data = pickle.load(pkl)
    rlist = data['rlist']
    zlist = data['zlist']
    counts, bin_edges = np.histogram(rlist, bins=30, normed=True)

    # # ------------------------SAVE RECON ERROR------------------------------
    plt.figure()
    plt.plot(bin_edges[1:], counts, linewidth=2)
    plt.xlabel(r"Distance, $\left \|\| I - \hat{I} \right \|\|$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
              fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('%s\\randomsearch.pdf' % path)
    plt.clf()
    plt.cla()
    plt.close()
    # # ------------------------SAVE LATENT ------------------------------
    zlist = np.concatenate(zlist)
    plt.figure()
    for i in range(zsize):
        plt.hist(zlist[:, i], bins='auto', histtype='step')
    plt.xlabel(r"$z$", fontsize=axis_title_size)
    plt.ylabel('Probability density', fontsize=axis_title_size)
    plt.title(r"PDF of embeding $p\left(z \right)$", fontsize=title_size)
    plt.grid(True)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout(rect=(0.0, 0.0, 1, 0.95))
    plt.savefig('%s\\embeding.pdf' % path)
    plt.clf()
    plt.cla()
    plt.close()
    gennorm_param = np.zeros([3, zsize])
    for i in range(zsize):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i])
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale
    return gennorm_param, bin_edges, counts, (G, E, D, ZD)


def compute_threshold(mnist_valid, percentage):
    # Searching for threshold on validation set
    random.shuffle(mnist_valid)
    mnist_valid_outlier = [x for x in mnist_valid if x[0] in outlier_classes]
    mnist_valid_inliner = [x for x in mnist_valid if x[0] in inliner_classes]

    inliner_count = len(mnist_valid_inliner)
    outlier_count = inliner_count * percentage // (100 - percentage)

    if len(mnist_valid_outlier) > outlier_count:
        mnist_valid_outlier = mnist_valid_outlier[:outlier_count]
    else:
        outlier_count = len(mnist_valid_outlier)
        inliner_count = outlier_count * (100 - percentage) // percentage
        mnist_valid_inliner = mnist_valid_inliner[:inliner_count]

    _mnist_valid = mnist_valid_outlier + mnist_valid_inliner
    random.shuffle(_mnist_valid)

    mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(_mnist_valid)
    if norm:
        mnist_valid_x = mnist_valid_x.reshape(-1, inputsize * inputsize)
        mnist_valid_x = normto01(mnist_valid_x)
    mnist_valid_x = mnist_valid_x.reshape(-1, inputsize, inputsize)

    result = []
    novel = []

    for it in range(len(mnist_valid_x) // batch_size):
        x = Variable(extract_batch(mnist_valid_x, it, batch_size).view(-1, inputsize, inputsize).data,
                     requires_grad=True)
        label = extract_batch_(mnist_valid_y, it, batch_size)
        z = E(x.view(-1, 1, inputsize, inputsize))
        recon_batch = G(z)
        z = z.squeeze()
        J = compute_jacobian(x, z)
        J = J.cpu().numpy()
        z = z.cpu().detach().numpy()
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        for i in range(batch_size):
            u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
            logD = np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
            p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
            logPz = np.sum(np.log(p))

            if not np.isfinite(logPz):
                logPz = -1000

            werror = ave_train * np.array(recon_batch[i].flatten() - x[i].flatten())
            distance = np.sum(np.power(werror, power))
            logPe = np.log(r_pdf(distance, bin_edges, counts))  # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            logPe -= np.log(distance) * (inputsize * inputsize - zsize)  # \| w^{\perp} \|}^{m-n}
            P = logD + logPz + logPe
            result.append(P)
            novel.append(label[i].item() in inliner_classes)

    result = np.asarray(result, dtype=np.float32)
    novel = np.asarray(novel, dtype=np.float32)
    minP = min(result) - 1
    maxP = max(result) + 1
    best_e = 0
    best_f = 0
    best_e_ = 0
    best_f_ = 0
    not_novel = np.logical_not(novel)

    for e in np.arange(minP, maxP, 0.1):
        y = np.greater(result, e)
        true_positive = np.sum(np.logical_and(y, novel))
        false_positive = np.sum(np.logical_and(y, not_novel))
        false_negative = np.sum(np.logical_and(np.logical_not(y), novel))

        if true_positive > 0:
            f = GetF1(true_positive, false_positive, false_negative)
            if f > best_f:
                best_f = f
                best_e = e
            if f >= best_f_:
                best_f_ = f
                best_e_ = e

    best_e = (best_e + best_e_) / 2.0
    logging.info('Best e:%f' % best_e)
    return best_e


def cal_acc(mnist_test_x, mnist_test_y, e):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    count = 0

    result = []
    batch_test = 16

    for it in range(len(mnist_test_x) // batch_test):
        x = Variable(extract_batch(mnist_test_x, it, batch_test).view(-1, inputsize, inputsize).data,
                     requires_grad=True)
        label = extract_batch_(mnist_test_y, it, batch_test)

        z = E(x.view(-1, 1, inputsize, inputsize))
        recon_batch = G(z)
        z = z.squeeze()

        J = compute_jacobian(x, z)

        J = J.cpu().numpy()

        z = z.cpu().detach().numpy()

        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        for i in range(batch_test):
            u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
            logD = np.sum(np.log(np.abs(s)))

            p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
            logPz = np.sum(np.log(p))

            if not np.isfinite(logPz):
                logPz = -1000

            werror = ave_train * np.array(recon_batch[i].flatten() - x[i].flatten())
            distance = np.sum(np.power(werror, power))

            logPe = np.log(r_pdf(distance, bin_edges, counts))
            logPe -= np.log(distance) * (inputsize * inputsize - zsize)

            count += 1

            P = logD + logPz + logPe

            if (label[i].item() in inliner_classes) != (P > e):
                if not label[i].item() in inliner_classes:
                    false_positive += 1
                if label[i].item() in inliner_classes:
                    false_negative += 1
            else:
                if label[i].item() in inliner_classes:
                    true_positive += 1
                else:
                    true_negative += 1

            result.append(((label[i].item() in inliner_classes), P))

    accuracy = 100 * (true_positive + true_negative) / count

    y_true = [x[0] for x in result]
    y_scores = [x[1] for x in result]

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0

    logging.info('accuracy :%f' % accuracy)
    f1 = GetF1(true_positive, false_positive, false_negative)
    logging.info('F1:%f' % f1)
    logging.info('AUC :%f' % auc)
    # inliers
    X1 = [x[1] for x in result if x[0]]

    # outliers
    Y1 = [x[1] for x in result if not x[0]]

    minP = min([x[1] for x in result]) - 1
    maxP = max([x[1] for x in result]) + 1

    # FPR at TPR 95
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    for e in np.arange(minP, maxP, 0.2):
        tpr = np.sum(np.greater_equal(X1, e)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr

    logging.info("tpr: ", clothest_tpr)
    logging.info("fpr95: ", fpr95)
    fdr = true_negative / (false_positive + true_negative)
    far = false_negative / (true_negative + false_negative)
    return fdr, far, f1, auc, fpr95, accuracy


def test(mnist_test, percentage, e):
    random.shuffle(mnist_test)
    mnist_test_outlier = [x for x in mnist_test if x[0] in outlier_classes]
    mnist_test_inliner = [x for x in mnist_test if x[0] in inliner_classes]

    inliner_count = len(mnist_test_inliner)
    outlier_count = inliner_count * percentage // (100 - percentage)

    if len(mnist_test_outlier) > outlier_count:
        mnist_test_outlier = mnist_test_outlier[:outlier_count]
    else:
        outlier_count = len(mnist_test_outlier)
        inliner_count = outlier_count * (100 - percentage) // percentage
        mnist_test_inliner = mnist_test_inliner[:inliner_count]

    mnist_test = mnist_test_outlier + mnist_test_inliner
    random.shuffle(mnist_test)

    mnist_test_x, mnist_test_y_all = list_of_pairs_to_numpy(mnist_test)
    if norm:
        mnist_test_x = mnist_test_x.reshape(-1, inputsize * inputsize)
        mnist_test_x = normto01(mnist_test_x)
    mnist_test_x_all = mnist_test_x.reshape(-1, inputsize, inputsize)
    total=[]
    f = open("%s\\acc.txt" % path, 'w')
    f.truncate()
    for k_class in outlier_classes:
        mnist_test_outlier = [x for x in mnist_test if x[0] in [k_class]]
        mnist_test_inliner = [x for x in mnist_test if x[0] in inliner_classes]
        mnist_test_new = mnist_test_outlier + mnist_test_inliner
        random.shuffle(mnist_test_new)
        mnist_test_x, mnist_test_y = list_of_pairs_to_numpy(mnist_test_new)
        if norm:
            mnist_test_x = mnist_test_x.reshape(-1, inputsize * inputsize)
            mnist_test_x = normto01(mnist_test_x)
        mnist_test_x = mnist_test_x.reshape(-1, inputsize, inputsize)
        acc = cal_acc(mnist_test_x, mnist_test_y, e)
        f.write('\n-----------------------------------class=%d------\n fdr = %f\n far = %f\n f1 = %f\n auc = %f\n fpr95 = %f\n accuracy = %f' % (
            k_class, acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]))
        total.append(acc)
    acc = cal_acc(mnist_test_x_all, mnist_test_y_all, e)
    f.write(
        '\n-----------------------------------all--------\n fdr = %f\n far = %f\n f1 = %f\n auc = %f\n fpr95 = %f\n accuracy = %f' % (
            acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]))
    f.close()
    total.append(acc)
    save = pd.DataFrame(total)
    save.to_csv("%s\\acc.csv" % path)


with open('process\\%s_fold_%d.pkl' % (datatpye, folding_id), 'rb') as pkl:
    mnist_test = pickle.load(pkl)
gennorm_param, bin_edges, counts, model = train_step(train_epoch)
e = compute_threshold(mnist_valid, 50)
test(mnist_test, 50, e)
