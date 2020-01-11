"""
controlledforests.py
Library of controlled forests

:link    http://cluo29.github.io/
:author  Chu Luo
:version v0.101 (11-Jan-2020)

Using some code from sklearn
(sklearn/ensemble/_forest.py) by Gilles Louppe <g.louppe@gmail.com>, Brian Holt <bdholt1@gmail.com>,
Joly Arnaud <arnaud.v.joly@gmail.com>, Fares Hedayati <fares.hedayati@gmail.com>

License: BSD 3 clause

"""


import numpy as np
import chuluodistribute as cld
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state


class ControlledForestClassifier:
    # controlled forests for classifier
    # it has a bunch of DecisionTreeClassifier in self.tree
    # initialization function
    def __init__(self,
                 n_estimators='warn',
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight

        self.tree = []
        self.all_distribution = None
        self.num_tree = n_estimators
        self.classes_ = []
        self.n_classes_ = []
        self.n_outputs_ = None

    def fit(self, feature_arrange, boot_x, boot_y, all_y):
        """Build a forest of trees using a given feature distribution set and bootstrap sample set
        Parameters
        ----------
        feature_arrange : feature distribution sets for each tree
            e.g., f1,f2,f3 for 2 trees
            [[f1,f2],[f2,f3]]
        boot_x : bootstrap sets of input samples
            e.g., X1,X2 for 2 trees, [[[1,2],[2,3]],[[1,2],[2,3]]]
            X1,X2 array-like or sparse matrix of shape = [n_samples, n_features]
        boot_y : bootstrap sets of output labels
            e.g., y1,y2 for 2 trees, [[1,0],[1,0]]
            y1 array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        all_y: original set of labels
        """

        self.all_distribution = feature_arrange


        # random_state follows generator and seed
        generator = check_random_state(self.random_state)

        # loop all the trees
        for i in range(self.n_estimators):
            # get the feature distribution of this tree
            current_distribution = list(feature_arrange[i])

            # need to sort the distribution before use
            current_distribution.sort()

            # get the all-feature trainset from all boot sets
            all_train_x = boot_x[i]

            train_y = boot_y[i]

            # make a train set using the features
            train_x = all_train_x[:, current_distribution]

            # train the tree
            # get a random number(will not have effects buy is needed)
            current_random_state = generator.randint(0, 0x7FFFFFFF)

            # make parameters correct
            # max_features is all (If None, then max_features=n_features)

            clf = DecisionTreeClassifier(criterion=self.criterion,
                                         splitter='best',
                                         max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                         max_features=None,
                                         random_state=current_random_state,
                                         max_leaf_nodes=self.max_leaf_nodes,
                                         min_impurity_decrease=self.min_impurity_decrease,
                                         min_impurity_split=self.min_impurity_split,
                                         class_weight=self.class_weight,
                                         presort=False)
            clf.fit(train_x, train_y, sample_weight=None, check_input=False)
            self.tree.append(clf)

        # save all classes
        y = np.atleast_1d(all_y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))
        y = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        self.n_outputs_ = y.shape[1]

        y_encoded = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_encoded[:, k] = np.unique(y[:, k],
                                                    return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]


    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)
        n_samples = X.shape[0]

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            predictions = np.zeros((n_samples, self.n_outputs_))

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(
                    np.argmax(proba[:, k], axis=1),
                    axis=0)

            return predictions

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class probabilities of the trees in the forest. The
        class probability of a single tree is the fraction of samples of the same
        class in a leaf.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """

        # get the feature distribution of this tree
        current_distribution = list(self.all_distribution[0])

        # need to sort the distribution before use
        current_distribution.sort()

        col_x0 = X[:, current_distribution]
        proba_x0 = self.tree[0].predict_proba(col_x0)

        proba_array = proba_x0 - proba_x0

        # for each tree, get proba using correct subsets of samples
        for i in range(self.num_tree):
            # get the feature distribution of this tree
            current_distribution = list(self.all_distribution[i])

            # need to sort the distribution before use
            current_distribution.sort()

            col_x_this = X[:, current_distribution]
            proba_x_this = self.tree[i].predict_proba(col_x_this)
            proba_array += proba_x_this

        proba_array /= self.num_tree

        return proba_array


class ControlledForestRegressor:
    # controlled forests for regression
    # it has a bunch of DecisionTreeRegressor in self.tree
    # initialization function
    def __init__(self,
                 n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

        self.tree = []
        self.all_distribution = None
        self.num_tree = n_estimators
        self.n_outputs_ = None


    def fit(self,feature_arrange,boot_x,boot_y,all_y):
        """Build a forest of trees using a given feature distribution set and bootstrap sample set
        Parameters
        ----------
        feature_arrange : feature distribution sets for each tree
            e.g., f1,f2,f3 for 2 trees
            [[f1,f2],[f2,f3]]
        boot_x : bootstrap sets of input samples
            e.g., X1,X2 for 2 trees, [[[1,2],[2,3]],[[1,2],[2,3]]]
            X1,X2 array-like or sparse matrix of shape = [n_samples, n_features]
        boot_y : bootstrap sets of output labels
            e.g., y1,y2 for 2 trees, [[1,0],[1,0]]
            y1 array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        all_y: original set of labels
        """

        self.all_distribution = feature_arrange

        # random_state follows generator and seed
        generator = check_random_state(self.random_state)

        # loop all the trees
        for i in range(self.n_estimators):
            # get the feature distribution of this tree
            current_distribution = list(feature_arrange[i])

            # need to sort the distribution before use
            current_distribution.sort()

            # get the all-feature trainset from all boot sets
            all_train_x = boot_x[i]

            train_y = boot_y[i]

            # make a train set using the features
            train_x = all_train_x[:, current_distribution]

            # train the tree
            # get a random number(will not have effects buy is needed)
            current_random_state = generator.randint(0, 0x7FFFFFFF)

            # make parameters correct
            # max_features is all (If None, then max_features=n_features)

            reg = DecisionTreeRegressor(criterion=self.criterion,
                                        splitter='best',
                                        max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                        max_features=None,
                                        random_state=current_random_state,
                                        max_leaf_nodes=self.max_leaf_nodes,
                                        min_impurity_decrease=self.min_impurity_decrease,
                                        min_impurity_split=self.min_impurity_split,
                                        presort=False)
            reg.fit(train_x, train_y, sample_weight=None, check_input=False)

            self.tree.append(reg)

        y = np.atleast_1d(all_y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

    def predict(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """
        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # count sum
        # for each tree
        for i in range(self.num_tree):
            col_x_this = X[:, self.all_distribution[i]]
            y_this = self.tree[i].predict(col_x_this)
            y_hat += y_this

        y_hat /= self.num_tree

        return y_hat

