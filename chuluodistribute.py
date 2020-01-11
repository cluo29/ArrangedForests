"""
chuluodistribute.py
Library of feature distribution algorithms DDA and MDDA

:link    http://cluo29.github.io/
:author  Chu Luo
:version v0.101 (11-Jan-2020)

Although code is totally written by Chu Luo,
the idea is proposed together by Chu Luo and Yuehui Zhang <zyh@sjtu.edu.cn> from Shanghai Jiao Tong University.

License: BSD 3 clause
"""

import numpy as np


def random_distribute(num_feature, num_trees, feature_per_tree, distributions, seed):
    """
    Randomly distribute features into trees, without replacement

    :param num_feature: number of features
    :param num_trees: number of trees
    :param feature_per_tree: number of features in one tree
    :param distributions: distributions already determined
    :param seed: random seed

    :return: the groups of features
        output example [[0,1] , [1,2]]
        feature id is from 0 to (num_feature-1)
    """
    # random distribution is quick, but not necessarily gives low similarity
    np.random.seed(seed)

    m = num_feature
    k = feature_per_tree

    tree_to_build = num_trees - len(distributions)

    # identify all features with array 0,1,2,...,num_feature-1

    # in set type, used internally
    all_distribution_set = []

    for i in distributions:
        all_distribution_set.append(set(i))

    final_distribution = []

    # randomly pick num_feature from all_feature
    # if it is in the set, try once more
    while len(all_distribution_set) < tree_to_build:

        random_set = set(np.random.choice(m, k, replace=False))

        if random_set not in all_distribution_set:
            all_distribution_set.append(random_set)
            if len(all_distribution_set) == tree_to_build:
                break

    for i in all_distribution_set:
        final_distribution.append(list(i))

    return final_distribution


# added 2019/10/02
def bootstrap_samples(num_trees, data_x, data_y, seed=1):
    """
    Randomly distribute samples into trees, with replacement

    :param num_trees: number of trees
    :param data_x: feature values
    :param data_y: labels
    :param seed: random seed

    :return: the groups of samples of feature values and labels
        labels at last column
        output example [[0,1] , [1,2]]

    """
    np.random.seed(seed)
    # np.random.choice
    # get sample row numbers
    num_instances = len(data_x)

    row_sets = []

    while len(row_sets) < num_trees:

        random_set = np.random.choice(num_instances, num_instances, replace=True)
        row_sets.append(random_set)

    # return num_trees sets of x and y

    boot_x = []
    boot_y = []

    for i in row_sets:
        boot_x.append(data_x[i])
        boot_y.append(data_y[i])

    return boot_x, boot_y


def build_feature_matrix(num_feature, feature_per_tree):
    """
    Build a feature matrix
    m must be k*k
    k = p

    :param num_feature: number of features
    :param feature_per_tree: number of features in one tree

    :return: k*k feature matrix A
        output example [[0,1] , [2,3]]
        feature id is from 0 to (num_feature-1)
    """
    m = num_feature
    k = feature_per_tree

    element_count = 0

    all_rows = []

    current_row = []

    for i in range(m):

        current_row.append(i)

        element_count += 1

        if element_count == k:
            all_rows.append(current_row)
            current_row = []
            element_count = 0

    feature_matrix = np.array(all_rows)

    return feature_matrix


def build_plus_feature_matrix(num_feature, feature_per_tree):
    """
    Build a feature matrix
    m must be k*k
    k + 1 = p

    e.g., build_plus_feature_matrix(25, 5)

    :param num_feature: number of features
    :param feature_per_tree: number of features in one tree

    :return: (k+1)*(k+1) feature matrix A+
        output example [[0,1] , [2,3]]
        feature id is from 0 to (num_feature-1)
        note: last row and column do not represent features
    """
    m = num_feature
    k = feature_per_tree

    element_count = 0

    all_rows = []

    current_row = []

    for i in range(m):

        current_row.append(i)

        element_count += 1

        if element_count == k:

            # for last column, item is 0

            current_row.append(0)

            all_rows.append(current_row)

            current_row = []

            element_count = 0

    # add last row, all 0

    all_rows.append(np.zeros((k+1,), dtype=int))

    feature_matrix = np.array(all_rows)

    return feature_matrix


def diagonal_distribute(feature_per_tree, feature_matrix):
    """
    Diagonal distribution algorithm (DDA)

    for k = p

    :param feature_per_tree: number of features in one tree, k
    :param feature_matrix: k*k feature matrix from 0 to m-1, A

    :return: the k family of features
        output example [[0,1] , [1,2]]
        feature id is from 0 to (num_feature-1)
    """
    k = feature_per_tree
    A = feature_matrix

    # k family subset to return, F
    k_family_F = []

    F1 = []

    for j in range(k):
        F1.append(list(A[j]))

    k_family_F = k_family_F + F1

    F2 = []

    for j in range(k):
        for l in range(k):
            T=[]
            for i in range(k):
                # mod k for all operations
                col_index = (j+i*l) % k
                T.append(A[i][col_index])
            F2.append(T)

    k_family_F = k_family_F + F2

    return k_family_F


def modified_diagonal_distribute(feature_per_tree, feature_matrix):
    """
    Modified diagonal distribution algorithm (MDDA)

    for k+1 = p

    :param feature_per_tree: number of features in one tree, k
    :param feature_matrix: k*k feature matrix from 0 to m-1, A

    :return: the k family of features
        output example [[0,1] , [1,2]]
        feature id is from 0 to (num_feature-1)
    """
    k = feature_per_tree
    A = feature_matrix

    # k family subset to return, F
    k_family_F = []

    F1 = []

    for j in range(k):
        F1.append(list(A[j]))

    k_family_F = k_family_F + F1

    # now time to use A+

    # get A+
    A_plus = build_plus_feature_matrix(k*k, k)

    F2 = []

    for j in range(k+1):
        for l in range(k+1):
            T = []
            for i in range(k+1):
                # mod k for all operations
                col_index = (j+i*l) % (k+1)

                if not(l == 0 and i == k) and not(l == 0 and j == k):
                    T.append([i, col_index])

            # check T for removing other non-feature items

            for indices in T:
                x = indices[0]
                y = indices[1]

                if x == k and y == k:
                    T.remove(indices)
                else:
                    if x == k:
                        # x_2, k in T
                        for indices2 in T:
                            if indices2[1] == k:
                                T.remove([k, y])
                                T.remove([indices2[0], k])
                                T.append([indices2[0], y])
                    elif y == k:
                        for indices2 in T:
                            if indices2[0] == k:
                                T.remove([x, k])
                                T.remove([k, indices2[1]])
                                T.append([x, indices2[1]])

            # append T indices in A+ one by one
            T_element = []
            for indices in T:
                T_element.append(A_plus[indices[0], indices[1]])

            if len(T_element) > 0:
                F2.append(T_element)

    k_family_F = k_family_F + F2

    return k_family_F


def get_repetition_index(distributions):
    """
    Count repetition index of a distribution arrangement

    :param distributions: distributions already determined

    :return: repetition index count
        output example 19
    """
    repetition_index = 0

    # check each pair without permutation

    num_distri = len(distributions)

    for i in range(num_distri-1):
        for j in range(i+1, num_distri):
            # get count between distributions[i], distributions[j]
            list_ij = distributions[i] + distributions[j]
            set_ij = set(list_ij)
            repetition_ij = len(distributions[i]) + len(distributions[j]) - len(set_ij)
            repetition_index += repetition_ij

    return repetition_index


def get_pair_repetition_index(distributions):
    """
    Count pairwise repetition index of a distribution arrangement

    :param distributions: distributions already determined

    :return: repetition index count matrix
        output example [[0,0],[0,0]]
    """

    num_distri = len(distributions)

    pair_repetition_index = np.zeros((num_distri, num_distri), int)

    for i in range(num_distri-1):
        for j in range(i+1, num_distri):
            # get count between distributions[i], distributions[j]
            list_ij = distributions[i] + distributions[j]
            set_ij = set(list_ij)
            repetition_ij = len(distributions[i]) + len(distributions[j]) - len(set_ij)
            pair_repetition_index[i, j] = repetition_ij

    return pair_repetition_index


def debug_modified_diagonal_distribute(feature_per_tree, feature_matrix):
    """
    DEBUG Modified diagonal distribution algorithm

    for k+1 = p

    :param feature_per_tree: number of features in one tree, k
    :param feature_matrix: k*k feature matrix from 0 to m-1, A

    :return: the k family of features
        output example [[0,1] , [1,2]]
        feature id is from 0 to (num_feature-1)
    """
    k = feature_per_tree
    A = feature_matrix

    # k family subset to return, F
    k_family_F = []

    F1 = []

    for j in range(k):
        F1.append(list(A[j]))

    k_family_F = k_family_F + F1

    # now time to use A+

    # get A+
    A_plus = build_plus_feature_matrix(k*k, k)

    F2 = []


    for j in range(k+1):
        for l in range(k+1):
            T = []
            for i in range(k+1):
                # mod k for all operations
                col_index = (j+i*l) % (k+1)

                if not(l == 0 and i == k) and not(l == 0 and j == k):
                    T.append([i, col_index])

            # check T for removing other non-feature items

            for indices in T:
                x = indices[0]
                y = indices[1]

                if x == k and y == k:
                    T.remove(indices)
                else:
                    if x == k:
                        # x_2, k in T
                        for indices2 in T:
                            if indices2[1] == k:
                                T.remove([k, y])
                                T.remove([indices2[0], k])
                                T.append([indices2[0], y])
                    elif y == k:
                        for indices2 in T:
                            if indices2[0] == k:
                                T.remove([x, k])
                                T.remove([k, indices2[1]])
                                T.append([x, indices2[1]])

            # append T indices in A+ one by one
            T_element = []
            for indices in T:
                T_element.append(A_plus[indices[0], indices[1]])

            if len(T_element) > 0:
                with open('output.txt', 'a') as f:
                    f.write('j = ' + str(j + 1))
                    f.write(', l = ' + str(l + 1))
                    f.write(', round 2 k family number: ' + str(len(F2)+1))
                    f.write(', items:  [')
                    for e in T_element:
                        f.write(str(e+1))
                        f.write(', ')
                    f.write(']\n')
                F2.append(T_element)




    k_family_F = k_family_F + F2

    return k_family_F


def debug2_modified_diagonal_distribute(feature_per_tree, feature_matrix):
    """
    DEBUG2 Modified diagonal distribution algorithm

    j and l loop replaced

    for k+1 = p

    :param feature_per_tree: number of features in one tree, k
    :param feature_matrix: k*k feature matrix from 0 to m-1, A

    :return: the k family of features
        output example [[0,1] , [1,2]]
        feature id is from 0 to (num_feature-1)
    """
    k = feature_per_tree
    A = feature_matrix

    # k family subset to return, F
    k_family_F = []

    F1 = []

    for j in range(k):
        F1.append(list(A[j]))

    k_family_F = k_family_F + F1

    # now time to use A+

    # get A+
    A_plus = build_plus_feature_matrix(k * k, k)

    F2 = []

    for l in range(k + 1):
        for j in range(k + 1):
            T = []
            for i in range(k + 1):
                # mod k for all operations
                col_index = (j + i * l) % (k + 1)

                if not (l == 0 and i == k) and not (l == 0 and j == k):
                    T.append([i, col_index])

            # check T for removing other non-feature items

            for indices in T:
                x = indices[0]
                y = indices[1]

                if x == k and y == k:
                    T.remove(indices)
                else:
                    if x == k:
                        # x_2, k in T
                        for indices2 in T:
                            if indices2[1] == k:
                                T.remove([k, y])
                                T.remove([indices2[0], k])
                                T.append([indices2[0], y])
                    elif y == k:
                        for indices2 in T:
                            if indices2[0] == k:
                                T.remove([x, k])
                                T.remove([k, indices2[1]])
                                T.append([x, indices2[1]])

            # append T indices in A+ one by one
            T_element = []
            for indices in T:
                T_element.append(A_plus[indices[0], indices[1]])

            if len(T_element) > 0:

                if l > 0:
                    with open('output13.txt', 'a') as f:
                        f.write('j = ' + str(j + 1))
                        f.write(', l = ' + str(l + 1))
                        f.write(', round 2 k family number: ' + str(len(F2) + 1))
                        f.write(', items:  [')
                        for e in T_element:
                            f.write(str(e + 1))
                            f.write(', ')
                        f.write(']\n')
                    F2.append(T_element)

    k_family_F = k_family_F + F2

    return F2


def debug_diagonal_distribute(feature_per_tree, feature_matrix):
    """
    Debug Diagonal distribution algorithm

    for k = p

    :param feature_per_tree: number of features in one tree, k
    :param feature_matrix: k*k feature matrix from 0 to m-1, A

    :return: the k family of features
        output example [[0,1] , [1,2]]
        feature id is from 0 to (num_feature-1)
    """
    k = feature_per_tree
    A = feature_matrix

    # k family subset to return, F
    k_family_F = []

    F1 = []

    for j in range(k):
        F1.append(list(A[j]))

    k_family_F = k_family_F + F1

    F2 = []

    for l in range(k):
        for j in range(k):
            T=[]
            for i in range(k):
                # mod k for all operations
                col_index = (j+i*l) % k
                T.append(A[i][col_index])
            F2.append(T)

            if 1 > 0:

                if l > 0:
                    with open('DDAoutput.txt', 'a') as f:
                        f.write('j = ' + str(j + 1))
                        f.write(', l = ' + str(l + 1))
                        f.write(', round 2 k family number: ' + str(len(F2) + 1))
                        f.write(', items:  [')
                        for e in T:
                            f.write(str(e + 1))
                            f.write(', ')
                        f.write(']\n')


    k_family_F = k_family_F + F2

    return F2
