from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import numpy as np


def welch_t_statistic(a: np.ndarray, b: np.ndarray) -> float:
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    return (mean_a - mean_b)/np.sqrt(np.sum(np.square(a-mean_a))/(len(a)-1)/len(a)+np.sum(np.square(b-mean_b))/(len(b)-1)/(len(b)))


def welch_t_statistic_all_cuts(arr, start, end):
    t = np.zeros(len(arr))
    mean_a = np.mean(arr[start-1:])
    mean_b = np.mean(arr[:start-1])
    sos_a = np.sum(np.square(arr[start-1:] - mean_a))
    sos_b = np.sum(np.square(arr[:start-1] - mean_b))
    for i_test in range(start, end):
        a = arr[i_test:]
        b = arr[:i_test]
        prev_mean_a = mean_a
        prev_mean_b = mean_b
        mean_a = mean_a - (arr[i_test-1]-mean_a)/(len(arr)-i_test)
        mean_b = mean_b + (arr[i_test-1]-mean_b)/i_test
        sos_a = sos_a - (arr[i_test-1]-mean_a)*(arr[i_test-1]-prev_mean_a)
        sos_b = sos_b + (arr[i_test-1]-mean_b)*(arr[i_test-1]-prev_mean_b)
        t[i_test] = (mean_a - mean_b)/np.sqrt(sos_a/(len(a)-1)/len(a)+sos_b/(len(b)-1)/(len(b)))
    return t


class D3:
    def __init__(self, window_size: int = 200, auc_threshold: float = 0.7, new_data_percentage: float = 0.5, stride=None, visualize=False, verbose=False, localize_drifts=True):
        """
        Discriminative Drift Detector from "Unsupervised Concept Drift Detection with a Discriminative Classifier" by Gözüaçık et al.
        :param window_size: how many samples are stored
        :param auc_threshold: how sensitive the drift detection is (higher value means fewer detections)
        :param new_data_percentage: how many samples from the window are labelled as new
        :param stride: how often D3 tests for drifts (e.g. 1 means each time a new sample arrives, "None" means as described in the paper)
        :param visualize: whether to do visualizations when a drift is detected
        :param verbose: whether to print additional information
        :param localize_drifts: whether to localize the drifts in time, as described in "Extending Drift Detection Methods to Identify When Exactly the Change Happened" by Vieth et al.
        """
        self.auc_threshold = max(min(auc_threshold, 1.0), 0.5)
        self.window_size = int(window_size/2)*2  # For now, we want an even integer
        self.new_data_window_size = int(self.window_size*new_data_percentage)
        if stride is None:
            self.stride = self.new_data_window_size
        else:
            self.stride = int(stride)
        self.data_window = None
        self.data_window_index = 0
        self.localize_drifts = localize_drifts
        self.drift_delay = self.new_data_window_size
        self.visualize = visualize
        self.verbose = verbose

    def update(self, sample) -> list:
        if self.data_window is None:  # in the constructor, we do not know how many features we are dealing with
            self.data_window = np.zeros((self.window_size, len(sample)))

        self.data_window[self.data_window_index] = sample
        self.data_window_index += 1

        if self.data_window_index < self.window_size:  # Not enough data yet
            return []

        y_true = np.zeros(self.window_size)
        y_true[-self.new_data_window_size:] = 1
        discriminative_classifier = LogisticRegression(random_state=0).fit(self.data_window, y_true)
        scores = discriminative_classifier.predict_proba(self.data_window)[:, 1]
        auc_score = roc_auc_score(y_true, scores)
        if self.verbose:
            print("sklearn auc score=", auc_score)
        if auc_score > self.auc_threshold:
            if self.localize_drifts:
                # determine exact drift point
                best_i = len(scores) - self.new_data_window_size
                best_i_value = -99999999.9
                mean_diffs = welch_t_statistic_all_cuts(scores, start=3, end=len(scores) - 2)
                for i_test in range(3, len(scores)-2):  # At least 3 samples on each side
                    mean_diff = mean_diffs[i_test]
                    if mean_diff > best_i_value:
                        best_i = i_test
                        best_i_value = mean_diff
                self.drift_delay = len(scores)-best_i-0.5
                if self.verbose:
                    print("best_i=", best_i, "best_i_value=", best_i_value, "drift_delay=", self.drift_delay)
                if self.visualize:
                    scores2 = np.full(self.window_size, fill_value=np.nan)
                    scores_raw = np.full(self.window_size, fill_value=np.nan)
                    correction_factor = np.full(self.window_size, fill_value=np.nan)
                    for i_test in range(3, len(scores)-2):  # At least 3 samples on each side
                        correction_factor[i_test] = 1.0/np.sqrt(np.var(scores[i_test:], ddof=1)/len(scores[i_test:])+np.var(scores[:i_test], ddof=1)/len(scores[:i_test]))
                        scores_raw[i_test] = np.mean(scores[i_test:]) - np.mean(scores[:i_test])
                        scores2[i_test] = ((np.mean(scores[i_test:]) - np.mean(scores[:i_test]))/np.sqrt(np.var(scores[i_test:], ddof=1)/len(scores[i_test:])+np.var(scores[:i_test], ddof=1)/(len(scores[:i_test]))))
                    fig1, ax1 = plt.subplots(layout="tight")
                    ax1.axvline(x=len(y_true)-self.new_data_window_size, color="tab:red", ls="--", label="initial D3 split")
                    ax1.axvline(x=len(y_true)-self.drift_delay, color="tab:red", label="actual drift location")
                    ax1.plot((scores_raw-np.nanmin(scores_raw))/(np.nanmax(scores_raw)-np.nanmin(scores_raw)), label="diff of means")
                    ax1.plot((scores2-np.nanmin(scores2))/(np.nanmax(scores2)-np.nanmin(scores2)), label="Welch's t-statistic")
                    ax1.plot(scores, "+", label="predicted sample probabilities")
                    ax1.legend()
                    ax1.set_xlabel("sample index")
                    ax1.set_ylabel("probability")
                    plt.show()

            self.data_window[:int(self.drift_delay), :] = self.data_window[-int(self.drift_delay):, :]
            self.data_window_index = int(self.drift_delay)
            return [self.drift_delay]
        else:
            self.data_window[:-self.stride, :] = self.data_window[self.stride:, :]
            self.data_window_index = self.data_window.shape[0]-self.stride
            return []
