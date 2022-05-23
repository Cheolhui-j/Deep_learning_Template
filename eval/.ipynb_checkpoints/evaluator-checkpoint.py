import math
import numpy as np
import time
import os
from ignite.metrics import Metric

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