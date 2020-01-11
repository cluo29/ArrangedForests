"""
arrangedforests.py
Library of arranged forests

:link    http://cluo29.github.io/
:author  Chu Luo
:version v0.101 (11-Jan-2020)

Although code is totally written by Chu Luo,
the idea is proposed together by Chu Luo and Yuehui Zhang <zyh@sjtu.edu.cn> from Shanghai Jiao Tong University.

License: BSD 3 clause
"""
from chuluodistribute import *
from controlledforests import *
from sklearn.metrics import classification_report


def example():
    # build arranged forests using ControlledForestClassifier with k-family and bootstrap

    # the dataset is from The UCI Machine Learning Repository
    # https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification

    # load data into numpy

    all_set = np.genfromtxt('pd_speech_features.csv', delimiter=',', skip_header=2)

    # all set includes train and test set

    print(all_set)

    row_all = len(all_set)

    print("row_all = ", row_all)

    column_all = np.size(all_set, 1)

    print("column_all = ", column_all)

    # delete the ID

    all_set = np.delete(all_set, 0, 1)

    column_all = np.size(all_set, 1)

    print("column_all = ", column_all)

    # use only 529 features
    columns_to_remove = range(529, 753)

    all_set = np.delete(all_set, columns_to_remove, 1)

    column_all = np.size(all_set, 1)

    print("column_all = ", column_all)

    # get first 528 rows (70%) as train set
    # the rest as test set

    train = all_set[0:528]
    test = all_set[528:]

    print(train)

    row_train = len(train)
    print("row_train = ", row_train)

    row_test = len(test)

    print("row_test = ", row_test)

    # get input and labels

    train_x = train[:, 0:-1]
    train_y = train[:, -1]

    test_x = test[:, 0:-1]
    test_y = test[:, -1]

    boot_train_x, boot_train_y = bootstrap_samples(552, train_x, train_y, seed=1)

    feature_matrix = build_feature_matrix(num_feature=529, feature_per_tree=23)

    k_family_F = diagonal_distribute(feature_per_tree=23, feature_matrix=feature_matrix)

    print(get_repetition_index(k_family_F))

    arranged_clf = ControlledForestClassifier(n_estimators=552)
    arranged_clf.fit(k_family_F, boot_train_x, boot_train_y, train_y)

    predict_y = arranged_clf.predict(test_x)

    arranged_report = classification_report(test_y, predict_y, output_dict=False)

    print(arranged_report)


example()
