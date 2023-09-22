import collections
import typing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import time
import random
import matplotlib.pyplot as plt


class SpectralDDM:
    @staticmethod
    def regularize_params(old_params: dict):
        new_params = {}
        for key in old_params:
            if key == "n_splits":
                new_params[key] = max(int(old_params[key]), 1)
            elif key == "test_size":
                new_params[key] = max(min(old_params[key], 0.95), 0.05)
            elif key == "min_samples_per_drift":
                new_params[key] = max(int(old_params[key]), 1)
            elif key == "n_eigen":
                new_params[key] = max(int(old_params[key]), 1)
            elif key == "max_possible_number_splits":
                new_params[key] = max(int(old_params[key]), 2)
            else:
                new_params[key] = old_params[key]
        return new_params

    def __init__(
        self,
        max_window_size: int = 500,
        min_window_size: int = 40,
        n_eigen: int = 10,
        max_possible_number_splits: int = 5,
        n_splits: int = 25,
        test_size: float = 0.45,
        min_samples_per_drift: int = 10,
        batch_distance: int = 75,
        visualize: bool = False,
        verbose: bool = False,
        localize_drifts=True,  # ignored
        stride=None  # ignored
    ):
        """
        Spectral Drift Detection Method from "Precise Change Point Detection using Spectral Drift Detection" by Hinder et al.
        :param max_window_size: how many samples are stored at most
        :param min_window_size: how many samples are collected before the first drift test is performed
        :param n_eigen: how many eigenvectors of L are used
        :param max_possible_number_splits: the maximum number of drifts in one window
        :param n_splits: number of cross-validation iterations
        :param test_size: portion used for testing in cross-validation
        :param min_samples_per_drift: minimum number of samples between two drifts (used for decision trees)
        :param batch_distance: how many samples are collected before the method tests for drifts again
        :param visualize: whether to visualize additional information
        :param verbose: whether to print additional information
        :param localize_drifts: ignored, this method always localizes drifts
        :param stride: ignored
        """
        if max_window_size < 1:
            raise ValueError("max_window_size must be greater than 0.")

        if min_window_size < 1:
            raise ValueError("min_window_size must be greater than 0.")

        if max_window_size < min_window_size:
            raise ValueError("max_window_size must be greater than min_window_size.")

        if n_eigen < 1:
            raise ValueError("n_eigen must be greater than zero.")

        if max_possible_number_splits < 2:
            raise ValueError("max_possible_number_splits must be at least 2.")

        if min_samples_per_drift < 1:
            raise ValueError("min_samples_per_drift must be greater than zero.")

        self.max_window_size = max(int(max_window_size), 10)
        self.min_window_size = max(int(min_window_size), 5)
        self.n_eigen = max(int(n_eigen), 1)
        self.max_possible_number_splits = max(int(max_possible_number_splits), 2)
        self.n_splits = max(int(n_splits), 1)
        self.test_size = max(min(test_size, 0.95), 0.05)
        self.min_samples_per_drift = max(int(min_samples_per_drift), 1)
        self.batch_distance = max(int(batch_distance), 1)
        self.detected_drifts = []  # to avoid/reduce duplicates
        self.visualize = visualize
        self.verbose = verbose

        self._reset()
        self.clock = self.batch_distance
        self.window = collections.deque(maxlen=self.max_window_size)

    def update(self, x):
        for i in range(len(self.detected_drifts)):
            self.detected_drifts[i] += 1

        self.window.append(x)  # add new sample
        if len(self.window) > self.max_window_size:
            self.window.popleft()
        if len(self.window) < self.min_window_size:
            return []
        if self.clock < self.batch_distance:
            self.clock += 1
            return []
        else:
            self.clock = 0
        X = np.zeros(shape=(len(self.window), len(x)))
        for i in range(len(self.window)):
            X[i, :] = self.window[i]
        K = pairwise_kernels(X, metric="rbf")
        assert K.shape[0] == len(self.window)
        assert K.shape[1] == len(self.window)
        D_prime = np.diag(np.divide(1.0, np.sqrt(np.sum(a=K, axis=0))))
        L = np.eye(N=K.shape[0]) - D_prime @ K @ D_prime
        eigenvals, eigenvecs = np.linalg.eigh(L)
        eigenvals = np.real(eigenvals)
        eigenvecs = np.real(eigenvecs)
        chosen_eigenvec = 0
        while eigenvals[chosen_eigenvec] < 1e-7:
            chosen_eigenvec += 1
        partitioning_tree_X = np.arange(0, len(self.window)).reshape(-1, 1)
        partitioning_tree_y = eigenvecs[:, chosen_eigenvec:(chosen_eigenvec+self.n_eigen)]
        num_leafs = []
        score = []
        scores = []
        num_leafs.append(1)
        scores.append(cross_val_score(estimator=DummyRegressor(strategy='mean'),
                                      X=partitioning_tree_X, y=partitioning_tree_y, scoring='r2',
                                      cv=ShuffleSplit(n_splits=self.n_splits,
                                                      test_size=min(partitioning_tree_X.shape[0]-1, int(np.ceil(self.test_size*partitioning_tree_X.shape[0]))),  # make sure that there is at least one sample for training (when test_size is close to 1 and number of sample (window size) is low)
                                                      random_state=42)))  # random state seed does not matter, but has to be the same for each call to ShuffleSplit!
        score.append(np.mean(scores[-1]))
        if self.verbose:
            print("1 leaf ", score[-1], ", split points: none")
        for number_of_leafs in range(2, self.max_possible_number_splits+1):
            cross_val_result = cross_val_score(estimator=DecisionTreeRegressor(max_leaf_nodes=number_of_leafs, min_samples_leaf=self.min_samples_per_drift),
                                               X=partitioning_tree_X, y=partitioning_tree_y, scoring='r2',
                                               cv=ShuffleSplit(n_splits=self.n_splits,
                                                               test_size=min(partitioning_tree_X.shape[0]-1, int(np.ceil(self.test_size*partitioning_tree_X.shape[0]))),
                                                               random_state=42))  # random state seed does not matter, but has to be the same for each call to ShuffleSplit!
            num_leafs.append(number_of_leafs)
            score.append(np.mean(cross_val_result))
            scores.append(cross_val_result)
            if self.verbose:
                partitioning_tree = DecisionTreeRegressor(max_leaf_nodes=number_of_leafs, min_samples_leaf=self.min_samples_per_drift).fit(X=partitioning_tree_X, y=partitioning_tree_y)
                split_points = np.sort(np.array(partitioning_tree.tree_.threshold[partitioning_tree.tree_.threshold > 0]))
                print(number_of_leafs, "leafs", score[-1], ", split points:", split_points)
        best_number_of_leafs = num_leafs[np.argmax(score)]

        if self.visualize:
            fig1, ax1 = plt.subplots()
            ax1.imshow(K)
            fig2, ax2 = plt.subplots()
            ax2.boxplot(scores)
            fig3, ax3 = plt.subplots()
            ax3.plot(eigenvecs[:, chosen_eigenvec:(chosen_eigenvec+self.n_eigen)])
            fig4, ax4 = plt.subplots()
            ax4.imshow((eigenvecs[:, chosen_eigenvec:(chosen_eigenvec+self.n_eigen)] @ np.diag(eigenvals[chosen_eigenvec:(chosen_eigenvec+self.n_eigen)]) @ eigenvecs[:, chosen_eigenvec:(chosen_eigenvec+self.n_eigen)].T))
            plt.show()
        
        if best_number_of_leafs > 1:
            partitioning_tree = DecisionTreeRegressor(max_leaf_nodes=best_number_of_leafs, min_samples_leaf=self.min_samples_per_drift).fit(X=partitioning_tree_X, y=partitioning_tree_y)
            split_points = np.sort(np.array(partitioning_tree.tree_.threshold[partitioning_tree.tree_.threshold > 0]))
            if self.verbose:
                print(best_number_of_leafs, "leaves, split points:", split_points)
            all_drift_delays = []
            for possible_drift in split_points:
                drift_delay = len(self.window) - 1 - possible_drift
                if self.verbose:
                    print(drift_delay, self.detected_drifts)
                already_reported = False
                for already_reported_drift in self.detected_drifts:
                    if abs(drift_delay-already_reported_drift)<self.min_samples_per_drift:
                        already_reported = True
                if not already_reported:
                    self.detected_drifts.append(drift_delay)
                    all_drift_delays.append(drift_delay)
                else:
                    if self.verbose:
                        print("already reported")
            return all_drift_delays
        else:
            if self.verbose:
                print("No splits")
            return []
