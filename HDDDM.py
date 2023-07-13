import numpy as np
from scipy.stats import t, ttest_ind, fisher_exact, barnard_exact, chi2_contingency
import matplotlib.pyplot as plt


def compute_histogram(X: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute a histogram from a collection of samples
    :param X: collection of samples, all elements must be between 0 and 1
    :param n_bins: number of bins for each dimension
    :return: the histogram
    """
    result = np.zeros((X.shape[1], n_bins))
    for i in range(X.shape[1]):
        result[i, :] = np.histogram(X[:, i], bins=n_bins, range=(0.0, 1.0), density=False)[0]
    return result


def eval_histogram(x, hist) -> float:  # assuming all elements of x are in [0; 1], and that the histogram is normalized (i.e. each row sums up to 1)
    """
    Determine how well a sample fits a histogram
    :param x: a sample
    :param hist: the histogram
    :return: a score specifying how well the sample fits the histogram
    """
    assert x.shape[0] == hist.shape[0]
    score = 0.0
    for i in range(x.shape[0]):
        score += hist[i, int(0.999*hist.shape[1]*x[i])]
    return score/x.shape[0]


def compute_hellinger_dist(P, Q):
    feature_distances = np.sqrt(np.sum(np.square(np.sqrt(np.divide(P, np.tile(np.sum(P, axis=1), (P.shape[1], 1)).transpose())) -
                                                 np.sqrt(np.divide(Q, np.tile(np.sum(Q, axis=1), (Q.shape[1], 1)).transpose()))), axis=1))
    return np.mean(feature_distances), feature_distances


class HDDDM:
    def __init__(self, gamma=None, alpha=None, batching_size=20, stride=None, visualize=False, verbose=False, localize_drifts=True):
        """
        Hellinger Distance Drift Detection Method from "Hellinger distance based drift detection for nonstationary environments" by Ditzler and Polikar
        :param gamma: how sensitive the drift detection is (higher value means fewer detections)
        :param alpha: a different way to specify how sensitive the drift detection is (either gamma or alpha must be specified, but not both)
        :param batching_size: the size of a batch (how many of the samples should make up the after-the-drift set)
        :param stride: currently unused
        :param visualize: whether to do visualizations when a drift is detected
        :param verbose: whether to print additional information
        :param localize_drifts: whether to localize the drifts in time, as described in "Extending Drift Detection Methods to Identify When Exactly the Change Happened" by Vieth et al.
        """
        if gamma is None and alpha is None:
            raise ValueError("Gamma and alpha can not be None at the same time! Please specify either gamma or alpha")
        elif gamma is not None and alpha is not None:
            raise ValueError("Specify either gamma or alpha, not both!")
        elif gamma is None and alpha is not None:
            self.gamma = None
            self.alpha = max(0.0, min(0.5, alpha))
        else:
            self.gamma = max(0.0, gamma)
            self.alpha = None
        self.batching_size = max(1, int(batching_size))
        if stride is None:
            self.stride = self.batching_size
        else:
            self.stride = int(stride)

        self.X_baseline = None
        self.n_bins = None
        self.hist_baseline = None
        self.n_samples = 0
        self.dist_old = np.nan
        self.epsilons = []
        self.accumulator = []
        self.drift_delay = self.batching_size
        self.localize_drifts = localize_drifts
        self.visualize = visualize
        self.verbose = verbose

        self.most_important_feature = 0
    
    def update(self, x):
        self.accumulator.append(x)
        if len(self.accumulator) >= self.batching_size:
            X = np.zeros(shape=(len(self.accumulator), len(self.accumulator[0])))
            for i in range(len(self.accumulator)):
                X[i, :] = self.accumulator[i]
            self.accumulator = []
            return self.add_batch(X)
        else:
            return []

    def add_batch(self, X):
        if self.n_bins is None:
            self.n_bins = int(np.floor(np.sqrt(X.shape[0])))
        if self.hist_baseline is None:
            self.X_baseline = X
            self.hist_baseline = compute_histogram(X, self.n_bins)
            self.n_samples = X.shape[0]
            return []

        hist = compute_histogram(X, self.n_bins)
        dist, all_feature_distances = compute_hellinger_dist(self.hist_baseline, hist)
        n_samples = X.shape[0]

        if np.isnan(self.dist_old):
            self.dist_old = dist
            self.hist_baseline += hist
            self.n_samples += n_samples
            self.X_baseline = np.vstack((self.X_baseline, X))
            return []
        eps = dist - self.dist_old
        self.dist_old = dist

        if len(self.epsilons) < 2:
            self.epsilons.append(eps)
            self.hist_baseline += hist
            self.n_samples += n_samples
            self.X_baseline = np.vstack((self.X_baseline, X))
            return []
        epsilon_hat = np.sum(np.abs(self.epsilons))/len(self.epsilons)
        sigma_hat = np.sqrt(np.sum(np.square(np.abs(self.epsilons) - epsilon_hat)) / len(self.epsilons))

        if self.gamma is not None:
            beta = epsilon_hat + self.gamma * sigma_hat
        else:
            beta = epsilon_hat + t.ppf(1.0 - self.alpha / 2, self.n_samples + n_samples - 2) * sigma_hat / np.sqrt(len(self.epsilons))
        self.epsilons.append(eps)

        # Test for drift
        if self.verbose:
            print("eps=", eps, "beta=", beta)
        drift = np.abs(eps) > beta

        if drift:
            if self.verbose:
                print("eps=", eps, "beta=", beta, "epsilon_hat=", epsilon_hat, "sigma_hat=", sigma_hat, "len(epsilons)=", len(self.epsilons), "epsilons=", self.epsilons)
            if self.localize_drifts:
                # determine drift location:
                scores_binary = []
                scores_cont = []
                hist_old = self.hist_baseline
                hist_new = hist
                hist_baseline_normalized = np.divide(hist_old, np.tile(np.sum(hist_old, axis=1), (hist_old.shape[1], 1)).transpose())
                hist_normalized = np.divide(hist_new, np.tile(np.sum(hist_new, axis=1), (hist_new.shape[1], 1)).transpose())
                for i in range(self.X_baseline.shape[0]):
                    a = eval_histogram(self.X_baseline[i, :], hist_baseline_normalized)
                    b = eval_histogram(self.X_baseline[i, :], hist_normalized)
                    scores_cont.append(b/(a+b))
                    scores_binary.append(a < b)
                for i in range(X.shape[0]):
                    a = eval_histogram(X[i, :], hist_baseline_normalized)
                    b = eval_histogram(X[i, :], hist_normalized)
                    scores_cont.append(b/(a+b))
                    scores_binary.append(a < b)
                best_i = len(scores_binary) - self.batching_size
                best_i_value = -99999999.9
                for i_test in range(max(3, len(scores_binary) - 2 * X.shape[0]), len(scores_binary)-2):
                    mean_diff = -fisher_exact([[np.sum(scores_binary[i_test:]), np.sum(scores_binary[:i_test])],
                                               [len(scores_binary[i_test:]) - np.sum(scores_binary[i_test:]), len(scores_binary[:i_test]) - np.sum(scores_binary[:i_test])]], alternative='greater').pvalue
                    if mean_diff > best_i_value:
                        best_i = i_test
                        best_i_value = mean_diff
                self.drift_delay = len(scores_binary) - best_i - 0.5
                if self.verbose:
                    print("best_i=", best_i, "best_i_value=", best_i_value, "len(scores_binary)=", len(scores_binary), "drift_delay=", self.drift_delay)

                if self.visualize:
                    a = np.zeros(len(scores_binary))
                    b = np.zeros(len(scores_binary))
                    correction = np.zeros(len(scores_binary))
                    for i_test in range(3, len(scores_binary)-2):  # At least 3 samples on each side
                        p2 = np.mean(scores_binary[:i_test])
                        p1 = np.mean(scores_binary[i_test:])
                        n2 = len(scores_binary[:i_test])
                        n1 = len(scores_binary[i_test:])
                        tmp = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
                        correction[i_test] = (p1 - p2) / np.sqrt(tmp)
                        a[i_test] = fisher_exact([[np.sum(scores_binary[:i_test]), len(scores_binary[:i_test])-np.sum(scores_binary[:i_test])], [np.sum(scores_binary[i_test:]), len(scores_binary[i_test:])-np.sum(scores_binary[i_test:])]], alternative='less').pvalue
                        b[i_test] = (np.mean(scores_cont[i_test:])-np.mean(scores_cont[:i_test]))

                    plt.axvline(x=len(scores_binary)-self.batching_size, color="tab:red", ls="--", label="initial histogram split")
                    plt.plot(scores_binary, label="binary scores")
                    plt.plot((scores_cont-np.nanmin(scores_cont))/(np.nanmax(scores_cont-np.nanmin(scores_cont))), label="continuous scores")
                    plt.plot(a, label="fisher_exact")
                    plt.plot(b/np.nanmax(b), label="uncorrected diff of means")
                    plt.plot(correction/np.nanmax(correction), label="correction")
                    plt.legend()
                    plt.show()

            self.epsilons = []
            self.dist_old = np.nan
            if self.drift_delay <= X.shape[0]:
                self.X_baseline = X[max(0, int(X.shape[0]-self.drift_delay)):, :]
            else:
                self.X_baseline = np.vstack((self.X_baseline[max(0, int(self.X_baseline.shape[0]-(self.drift_delay-X.shape[0]))):, :], X))
            self.n_samples = self.X_baseline.shape[0]
            self.n_bins = int(np.floor(np.sqrt(self.batching_size)))
            self.hist_baseline = compute_histogram(self.X_baseline, self.n_bins)
            return [self.drift_delay]
        else:
            self.hist_baseline += hist
            self.n_samples += n_samples
            self.X_baseline = np.vstack((self.X_baseline, X))
            return []
