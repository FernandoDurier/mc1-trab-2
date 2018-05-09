import numpy as np
import scipy
from statsmodels.stats.weightstats import \
                DescrStatsW, CompareMeans, ttest_ind, ztest, zconfint

def kolgomorov2samples(dist1,dist2):
    return scipy.stats.ks_2samp(dist1,dist2)

def testz(dist1,dist2):
    return ztest(dist1, dist2, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)


