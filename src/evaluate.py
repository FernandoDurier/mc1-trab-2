import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from preprocess import p_method, n_method
from tests import kolgomorov2samples, testz


def sample_crowd(dataset, size):
    samples_idx = np.random.choice(dataset.shape[0], size=size, replace=False)
    return dataset[samples_idx, :]

def correct(gs, mean):
    gs = gs.astype(np.float64)
    indexes = (gs[:, 1] > mean).A1
    y_true = np.zeros(42).astype(np.int)
    for index, correct in enumerate((gs[:, 1] > mean).A1):
        if correct:
            y_true[index] = 1

    return y_true

def predict(proposals):
    return [1 if i in proposals else 0 for i in range(0, 42)]

def metrics(dataset, y_true, crowd_size, sample_size=100):
    precision = np.zeros((sample_size, 3))
    recall = np.zeros((sample_size, 3))
    f_score = np.zeros((sample_size, 3))
    for run in range(0, sample_size):
        sampled_dataset = sample_crowd(dataset, crowd_size)
        p_n, r_n, f_n, _ = precision_recall_fscore_support(
            y_true,
            predict(n_method(sampled_dataset, 21)), # 21 is 0.5 * 42 (compression ratio * #proposals)
            average='macro',
        )
        p_p, r_p, f_p, _ = precision_recall_fscore_support(
            y_true,
            predict(p_method(sampled_dataset, 21)), # 21 is 0.5 * 42 (compression ratio * #proposals)
            average='macro',
        )
        precision[run] = np.array([run, p_n, p_p])
        recall[run] = np.array([run, r_n, r_p])
        f_score[run] = np.array([run, f_n, f_p])

    return precision, recall, f_score 


def hypothesis_tests(metric_matrix):
    nonparametric = kolgomorov2samples(metric_matrix[:, 1], metric_matrix[:, 2])
    parametric = testz(metric_matrix[:, 1], metric_matrix[:, 2])
    return np.array([
        'rejects' if nonparametric.pvalue <= 0.05 else 'cannot reject',
        f'{nonparametric.pvalue:.3f}',
        'rejects' if parametric[1] <= 0.05 else 'cannot reject',
        f'{parametric[1]:.3f}',
    ])
