import math
import numpy as np
import time
import os
from ignite.metrics import Metric

import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate

class ROC(Metric):
    
    def __init__(self):
        super(ROC, self).__init__()
        self.reset()

    def reset(self):
        self.hist_genuine = np.zeros(100001)
        self.hist_imposter = np.zeros(100001)
        self.total_genuine = 0
        self.total_imposter = 0

    def update(self, output):

        embeddings, ids = output

        embeddings = embeddings.cpu().numpy()
        ids = ids.cpu().numpy()

        # ============================================
        # calculate cross matching scores and stack as a histogram
        # ============================================
        for i in range(embeddings.shape[0]):
            for j in range(i):
                sum_diff = 0
                for k in range(32):
                    sum_diff += math.pow(embeddings[j, k] - embeddings[i, k], 2)
                score = 100000 - (sum_diff * (100000/(4)))

                if ids[j] == ids[i]: # genuine case
                    self.hist_genuine[int(score)] += 1
                    self.total_genuine += 1
                else: # imposter case
                    self.hist_imposter[int(score)] += 1
                    self.total_imposter += 1


    def compute(self):

        hist_scale = 100000

        cum_genuine = 0
        cum_imposter = 0

        thresholds = np.arange(hist_scale, 0, -1)
        fars = [0 for i in range(len(thresholds) + 1)]
        frrs = [0 for i in range(len(thresholds) + 1)]

        eer_count = 0
        eer = None
        roc_result = ''

        # ============================================
        # calculate ROC curve (FAR, FRR, EER)
        # ============================================
        for threshold_idx, threshold in enumerate(thresholds):
            threshold = threshold.tolist()
            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.000000001) and (float(cum_imposter/self.total_imposter) < 0.000000001):
                roc_result += "\nFRR {0:6.3f}%, @ FAR9, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.00000001) and (float(cum_imposter/self.total_imposter) < 0.00000001):
                roc_result += "FRR {0:6.3f}%, @ FAR8, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.0000001) and (float(cum_imposter/self.total_imposter) < 0.0000001):
                roc_result += "FRR {0:6.3f}%, @ FAR7, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.000001) and (float(cum_imposter/self.total_imposter) < 0.000001):
                roc_result += "FRR {0:6.3f}%, @ FAR6, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.00001) and (float(cum_imposter/self.total_imposter) < 0.00001):
                roc_result += "FRR {0:6.3f}%, @ FAR5, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.0001) and (float(cum_imposter/self.total_imposter) < 0.0001):
                roc_result += "FRR {0:6.3f}%, @ FAR4, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            if ((float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter) >= 0.001) and (float(cum_imposter/self.total_imposter) < 0.001):
                roc_result += "FRR {0:6.3f}%, @ FAR3, (Threshold = {1:.5f})\n".format(100 * (self.total_genuine - cum_genuine)/self.total_genuine, threshold / hist_scale)

            fars[threshold] = float(cum_imposter + self.hist_imposter[threshold])/self.total_imposter
            frrs[threshold] = float(self.total_genuine - cum_genuine)/self.total_genuine

            if eer_count == 0:
                if (abs(fars[threshold] - frrs[threshold]) < 0.000000001):
                    eer = frrs[threshold]
                    eer_threshold = threshold
                    eer_count = 1
                elif (abs(fars[threshold] - frrs[threshold]) < 0.00000001):
                    eer = frrs[threshold]
                    eer_threshold = threshold
                    eer_count = 1
                elif (abs(fars[threshold] - frrs[threshold]) < 0.0000001):
                    eer = frrs[threshold]
                    eer_threshold = threshold
                    eer_count = 1
                elif (abs(fars[threshold] - frrs[threshold]) < 0.000001):
                    eer = frrs[threshold]
                    eer_threshold = threshold
                    eer_count = 1
                elif (abs(fars[threshold] - frrs[threshold]) < 0.00001):
                    eer = frrs[threshold]
                    eer_threshold = threshold
                    eer_count = 1
                elif (abs(fars[threshold] - frrs[threshold]) < 0.0001):
                    eer = frrs[threshold]
                    eer_threshold = threshold
                    eer_count = 1
                else:
                    eer_count = 0

            cum_genuine += self.hist_genuine[threshold]
            cum_imposter += self.hist_imposter[threshold]

        if eer_count == 1:
            roc_result += "EER {0:6.3f}%, (Threshold = {1:.5f})\n\n".format(100 * eer, eer_threshold / hist_scale)
        if eer_count == 0:
            roc_result += "Cannot calcuate EER\n\n"

        roc_result += "Total count = {:,}\n".format(self.total_genuine + self.total_imposter)
        roc_result += "Total genuine count = {:,}\n".format(self.total_genuine)
        roc_result += "Total imposter count = {:,}\n".format(self.total_imposter)

        return roc_result

def pair_matching_accuracy(model, carray, issame, emd_size, device):
    """ Overall pair matching accuracy calculation process """

    st = time.time()

    # ============================================
    # mode as eval mode
    # ============================================
    model.eval()

    # ============================================
    # compute the embedding features from all test images
    # ============================================
    idx = 0
    batch_size = 512
    val_data_num = len(carray)
    embeddings = np.zeros([val_data_num, emd_size])

    with torch.no_grad():
        while idx + batch_size <= val_data_num:
            batch = torch.tensor(carray[idx:idx + batch_size])
            embeddings[idx:idx + batch_size] = F.normalize(model(batch.to(device))).cpu()
            idx += batch_size
        if idx < val_data_num:
            batch = torch.tensor(carray[idx:])
            embeddings[idx:] = F.normalize(model(batch.to(device))).cpu()

    # ============================================
    # calculate the performance
    # ============================================
    acc_result = ''
    tpr, fpr, accuracy, best_thresholds = calculate_verification_performance(embeddings, issame, nrof_folds=5, pca=0)
    acc_result += 'Accuracy: {:.5f}\n'.format(accuracy.mean())
    print('Accuracy: {:.5f}'.format(accuracy.mean()))
    acc_result += 'Total image pairs: {:,}\n'.format(int(val_data_num/2))
    print('Total image pairs: {:,}'.format(int(val_data_num/2)))

    # ============================================
    # compute the inference time
    # ============================================
    inf_time = time.time() - st
    acc_result += 'infer time {:.6f} sec\n'.format(inf_time)
    print('infer time {:.6f} sec\n'.format(inf_time))

    # ============================================
    # mode back to train mode
    # ============================================
    model.train()

    return accuracy.mean(), inf_time, acc_result

def calculate_verification_performance(embeddings, actual_issame, nrof_folds=10, pca=0):

    # ============================================
    # split embedding features to make pairs
    # ============================================
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    # ============================================
    # split embedding features to make pairs
    # ============================================
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    thresholds = np.arange(0, 4, 0.01)
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    # ============================================
    # calculate the euclidean distance of image pairs (not using pca)
    # ============================================
    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    # ***elif
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # ============================================
        # calculate the euclidean distance of image pairs (pca)
        # ============================================
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)

            # compute PCA
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)

            # calculate dist
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # ============================================
        # find the best threshold for the fold
        # ============================================

        # caculate trainset accuracy
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])

        # find the best threshold for trainset
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]

        # caculate testset accuracy
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

def calculate_accuracy(threshold, dist, actual_issame):
    """ Calculate the evaluation metrics (tpr, fpr, accuracy) """

    # ============================================
    # prediction by thresholding the distance
    # ============================================
    predict_issame = np.less(dist, threshold)

    # ============================================
    # calculate tp, fp, tn, fn
    # ============================================
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    acc = float(tp + tn) / dist.size

    return tpr, fpr, acc