from random import randint
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_kernels
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
import matplotlib.pyplot as plt


def MMD2u(K: np.ndarray, m: int, n: int) -> float:
    """
    Compute the unbiased Maximum Mean Discrepancy score
    :param K: square, symmetric kernel matrix (size should be (n+m) times (n+m))
    :param m: size of the first sample set
    :param n: size of the second sample set
    :return: MMD score
    """
    assert K.shape[0] == K.shape[1]
    assert K.shape[0] == (m+n)
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


def correctedMMD2u(K: np.ndarray, m: int, n: int) -> float:
    """
    An adaptation of the unbiased MMD, similar to the t-statistic from Welch's t-test
    :param K: square, symmetric kernel matrix (size should be (n+m) times (n+m))
    :param m: size of the first sample set
    :param n: size of the second sample set
    :return: "corrected" MMD score
    """
    assert K.shape[0] == K.shape[1]
    assert K.shape[0] == (m+n)
    mean1 = (np.sum(K[:m, :m]) - np.sum(np.diag(K[:m, :m]))) / (m * m - m)
    mean2 = (np.sum(K[m:, m:]) - np.sum(np.diag(K[m:, m:]))) / (n * n - n)
    mean3 = np.sum(K[:m, m:]) / (m * n)
    var1 = np.sum(np.square(np.tril(K[:m, :m] - mean1, k=-1))) / ((m*m-m) / 2 - 1)
    var2 = np.sum(np.square(np.tril(K[m:, m:] - mean2, k=-1))) / ((n*n-n) / 2 - 1)
    var3 = np.sum(np.square(K[:m, m:]-mean3)) / (m*n - 1)
    tmp = var1/(m*m-m) + var2/(n*n-n) + 2.0 * var3/(m*n)
    return (mean1 + mean2 - 2.0 * mean3) / np.sqrt(tmp)


def correctedMMD2u_all_cuts(K: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Compute the "corrected" unbiased MMD for different separations into two sample sets. Is fast by doing incremental updates of the means and variances
    :param K: square, symmetric kernel matrix (size should be (n+m) times (n+m))
    :param start: start point of the cuts into two sample sets
    :param end: end point of the cuts into two sample sets
    :return: 1D array with the scores
    """
    res = np.zeros(K.shape[0])
    m = start-1
    n = K.shape[0]-m
    mean1 = (np.sum(K[:m, :m]) - np.sum(np.diag(K[:m, :m]))) / (m * m - m)
    mean2 = (np.sum(K[m:, m:]) - np.sum(np.diag(K[m:, m:]))) / (n * n - n)
    mean3 = np.sum(K[:m, m:]) / (m * n)
    sos1 = np.sum(np.square(np.tril(K[:m, :m] - mean1, k=-1)))
    sos2 = np.sum(np.square(np.tril(K[m:, m:] - mean2, k=-1)))
    sos3 = np.sum(np.square(K[:m, m:] - mean3))
    for i in range(start, end):
        m = i
        n = K.shape[0]-i
        prev_mean1 = mean1
        prev_mean2 = mean2
        prev_mean3 = mean3
        mean1 = mean1+2*(np.sum(K[m-1, :m-1])-mean1*(m-1))/(m*(m-1))
        if i == end-10:  # refresh (accumulated errors)
            mean2 = (np.sum(K[m:, m:]) - np.sum(np.diag(K[m:, m:]))) / (n * n - n)
        else:
            mean2 = mean2+2*(mean2*n-np.sum(K[m-1, m:]))/(n*(n-1))
        mean3 = mean3+(mean3*(m-n-1)-np.sum(K[m-1, :m-1])+np.sum(K[m-1, m:]))/(m*n)
        sos1 = sos1 + np.sum((K[m-1, :m-1]-mean1)*(K[m-1, :m-1]-prev_mean1))
        sos2 = sos2 - np.sum((K[m-1, m:]-mean2)*(K[m-1, m:]-prev_mean2))
        sos3 = sos3 - np.sum((K[m-1, :m-1]-mean3)*(K[m-1, :m-1]-prev_mean3)) + np.sum((K[m-1, m:]-mean3)*(K[m-1, m:]-prev_mean3))
        res[i] = (mean1 + mean2 - 2.0 * mean3) / np.sqrt(sos1/((m*m-m)/2-1)/(m*m-m) + sos2/((n*n-n)/2-1)/(n*n-n) + 2.0 * sos3/(m*n-1)/(m*n))
    return res



def gauss_kernel_from_distances(pd: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Given a square, symmetric matrix of sample-to-sample distances, compute the element-wise gauss kernel
    :param pd: square, symmetric matrix of sample-to-sample distances
    :param sigma2: variance parameter for gauss kernel
    :return: matrix of same shape as pd
    """
    if sigma2 < 1e-10:
        sigma2 = 1e-10
    return np.exp((-1.0 / sigma2) * pd)


def varnodiag(arr, ddof):  # variance of a symmetric array, not including the diagonal
    """

    :param arr:
    :param ddof:
    :return:
    """
    assert arr.shape[0] == arr.shape[1]
    count = (arr.shape[0]**2-arr.shape[0])/2
    mean = (np.sum(arr)-np.sum(np.diag(arr)))/(arr.shape[0]**2-arr.shape[0])
    var = np.sum(np.square(np.tril(arr-mean, k=-1)))/(count-ddof)
    return var


class MMDDDM:
    def __init__(self, gamma, alpha=None, use_k2s_test=False, batching_size=20, stride=None, max_history=1000, visualize=False, verbose=False, localize_drifts=True):
        """
        Maximum Mean Discrepancy Drift Detection Method
        :param gamma:
        :param alpha:
        :param use_k2s_test:
        :param batching_size: how many of the samples should make up the after-the-drift set
        :param stride: how often MMDDDM tests for drifts (e.g. 1 means each time a new sample arrives, "None" means same as batching size)
        :param max_history: store at most this many samples
        :param visualize: whether to do visualizations when a drift is detected
        :param verbose: whether to print additional information
        :param localize_drifts: whether to localize the drifts in time, as described in "Extending Drift Detection Methods to Identify When Exactly the Change Happened" by Vieth et al.
        """
        if gamma is None and alpha is None:
            raise ValueError("Gamma and alpha can not be None at the same time! Please specify either gamma or alpha")

        self.use_k2s_test = use_k2s_test

        self.gamma = max(0.0, gamma)
        self.alpha = alpha
        self.batching_size = int(batching_size)
        if stride is None:
            self.stride = self.batching_size
        else:
            self.stride = int(stride)
        self.max_history = int(max_history)

        self.dist_old = 0.
        self.epsilons = []
        self.clock_since_last_test = self.stride
        self.X_baseline = None
        self.pair_dists = np.zeros((0, 0))
        self.last_K = np.zeros((0, 0))
        self.last_K_sigma2 = np.nan
        self.drift_delay = self.batching_size
        self.localize_drifts = localize_drifts
        self.visualize = visualize
        self.verbose = verbose

    def update(self, x) -> list:
        x = x.reshape(1, x.size)
        assert x.ndim == 2

        if self.X_baseline is None:
            self.X_baseline = x
            self.pair_dists = np.zeros((1, 1))
            self.last_K = np.ones((1, 1))
            return []

        self.X_baseline = np.vstack([self.X_baseline, x])
        if self.X_baseline.shape[0] < 2 * self.batching_size:
            return []
        self.clock_since_last_test += 1
        if self.clock_since_last_test < self.stride:
            return []
        self.clock_since_last_test = 0
        count_new_data = self.X_baseline.shape[0] - self.pair_dists.shape[0]
        self.pair_dists = np.pad(self.pair_dists, pad_width=((0, count_new_data), (0, count_new_data)))
        self.pair_dists[:-count_new_data, -count_new_data:] = pairwise_distances(self.X_baseline[:-count_new_data, :], self.X_baseline[-count_new_data:, :], metric='euclidean')
        self.pair_dists[-count_new_data:, :-count_new_data] = self.pair_dists[:-count_new_data, -count_new_data:].transpose()
        if count_new_data > 1:
            self.pair_dists[-count_new_data:, -count_new_data:] = pairwise_distances(self.X_baseline[-count_new_data:, :], metric='euclidean')
        else:
            self.pair_dists[-1, -1] = 0.0
        subset = np.random.choice(self.pair_dists.flat, size=min(self.pair_dists.size, 999), replace=True)
        if np.isnan(self.last_K_sigma2):
            sigma2 = np.median(subset) ** 2
            self.last_K = gauss_kernel_from_distances(self.pair_dists, sigma2)
            self.last_K_sigma2 = sigma2
        else:
            bounds = np.square(np.percentile(subset, q=[47, 53]))
            if bounds[0] < self.last_K_sigma2 < bounds[1]:
                self.last_K = np.pad(self.last_K, pad_width=((0, count_new_data), (0, count_new_data)))
                self.last_K[:-count_new_data, -count_new_data:] = gauss_kernel_from_distances(self.pair_dists[:-count_new_data, -count_new_data:], self.last_K_sigma2)
                self.last_K[-count_new_data:, :-count_new_data] = self.last_K[:-count_new_data, -count_new_data:].transpose()
                if count_new_data > 1:
                    self.last_K[-count_new_data:, -count_new_data:] = gauss_kernel_from_distances(self.pair_dists[-count_new_data:, -count_new_data:], self.last_K_sigma2)
                else:
                    self.last_K[-1, -1] = 1.0
            else:
                sigma2 = np.median(subset) ** 2
                self.last_K = gauss_kernel_from_distances(self.pair_dists, sigma2)
                self.last_K_sigma2 = sigma2
        K = self.last_K
        dist = MMD2u(K, K.shape[0]-self.batching_size, self.batching_size)

        drift = dist > self.gamma

        if drift:
            if self.verbose:
                print("drift detected, dist=", dist, "threshold=", self.gamma)
            if self.localize_drifts:
                all_MMDs = correctedMMD2u_all_cuts(K, max(3, K.shape[0]-self.stride-self.batching_size), K.shape[0]-2)
                best_i = K.shape[0] - self.batching_size
                best_i_value = -99999999.9
                for i in range(max(3, K.shape[0]-self.stride-self.batching_size), K.shape[0]-2):
                    curr_value = all_MMDs[i]
                    if curr_value > best_i_value:
                        best_i_value = curr_value
                        best_i = i
                self.drift_delay = K.shape[0]-best_i - 0.5
                if self.verbose:
                    print("best_i=", best_i, "best_i_value=", best_i_value, "K.shape=", K.shape, "drift_delay=", self.drift_delay)

                if self.visualize:
                    scores = np.zeros(K.shape[0])
                    scores2 = np.zeros(K.shape[0])
                    penalty = np.zeros(K.shape[0])
                    for i in range(3, K.shape[0]-2):
                        penalty[i] = 1/np.sqrt(varnodiag(K[:i, :i], ddof=1)/(i**2-i)+varnodiag(K[i:, i:], ddof=1)/((K.shape[0]-i)**2-(K.shape[0]-i))+2.0*np.var(K[:i, i:], ddof=1)/(i*(K.shape[0]-i)))
                        scores[i] = MMD2u(K, i, K.shape[0]-i)*penalty[i]
                        scores2[i] = MMD2u(K, i, K.shape[0]-i)
                    fig1, ax1 = plt.subplots()
                    fig1.tight_layout()
                    ax1.imshow(K)
                    ax1.axhline(y=K.shape[0]-self.batching_size, color="tab:red")
                    ax1.axvline(x=K.shape[0]-self.batching_size, color="tab:red")
                    fig2, ax2 = plt.subplots(layout="tight")
                    ax2.axhline(y=self.gamma/np.nanmax(scores2))
                    ax2.axvline(x=K.shape[0]-self.batching_size, color="tab:red", ls="--", label="initial MMD split")
                    ax2.axvline(x=K.shape[0]-self.drift_delay, color="tab:red", label="actual drift location")
                    ax2.plot(scores2/np.nanmax(scores2), label="uncorrected MMD")
                    ax2.plot(scores/np.nanmax(scores), label="corrected MMD")
                    ax2.legend()
                    ax2.set_xlabel("sample index")
                    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    ax2.set_xlim([0, K.shape[0]+1])
                    plt.show()

            self.epsilons = []
            self.X_baseline = self.X_baseline[-int(self.drift_delay):, :]
            self.pair_dists = self.pair_dists[-int(self.drift_delay):, -int(self.drift_delay):]
            self.last_K = self.last_K[-int(self.drift_delay):, -int(self.drift_delay):]
            assert self.pair_dists.shape[0] == self.pair_dists.shape[1]
            assert self.pair_dists.shape[0] == self.X_baseline.shape[0]
            return [self.drift_delay]
        else:
            assert self.pair_dists.shape[0] == self.pair_dists.shape[1]
            assert self.pair_dists.shape[0] == self.X_baseline.shape[0]
            return []
