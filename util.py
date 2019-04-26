import random
import time
from random import randrange

import numpy as np
import torch


def init_weights(module, mean=0, std=0.01):
    '''
    Utility method for initializing value of weights and biases by normal distribution.
    :param module: given module to apply changes
    :param mean: mean of normal distribution
    :param std: standard deviation of normal distribution
    :return: void
    '''
    if type(module) == torch.nn.Linear:
        module.weight.data.normal_(mean, std)
        module.bias.data.normal_(mean, std)
    elif type(module) == torch.nn.Embedding:
        module.weight.data.normal_(mean, std)


def get_data(filename, delimiter):
    '''
    Reads first two column of data and finds maximum of each column(user, item) and creates a dict,
    which keys are users and values are set of interacted items.
    :return: dict, num_users, num_items
    '''
    data = np.loadtxt(filename, dtype=int, usecols=(0, 1), delimiter=delimiter)

    # Plus 1 because data is 1 based(used in embedding sizes.
    [num_users, num_items] = np.max(data, axis=0) + 1

    indexed_data = {}
    for [user, item] in data:
        indexed_data[user] = indexed_data.get(user, set()) | {item}

    return indexed_data, num_users, num_items


def get_test_set(data, num_users):
    '''
    create test set by leave-one-out evaluation approach, i'th index of test set indicates test item
    of i'th user.
    :return: test set as array
    '''
    test_set = [0] * num_users
    for user in data.keys():
        items = data[user]
        test_set[user] = random.sample(items, 1)[0]

    return test_set


def get_train_set(data, test_set, num_items, num_negative_samples, batch_size):
    '''
    Creates train set with specification of paper which samples num_negative_samples non interacted
    samples per interacted sample. and shuffles it convert it to a numpy array which have 3 array
    containing users, items and labels which are 1 for interacted samples and 0 o.w. and then split
    it to have batches with specified batch_size.
    :return: numpy array train set
    '''
    print("preparing train set...")
    t = time.time()
    train_set = []
    for user in data.keys():
        items = data[user]
        # Add interacted user-items.
        for item in items:
            # we must exclude test items.
            if item != test_set[user]:
                train_set.append([user, item, 1])

        # Add num_negative_samples(4) non interacted user-items per each interacted user-item.
        for i in range(num_negative_samples * len(items)):
            rand_item = randrange(1, num_items)
            while rand_item in items:
                rand_item = randrange(1, num_items)

            train_set.append([user, rand_item, 0])

    # Convert list to numpy array.
    train_set = np.asarray(train_set, dtype=int)

    np.random.shuffle(train_set)

    # Transposing np array to all users, items, and labels come together.
    train_set = train_set.T

    # Splitting array to get list of batches with given batch size.
    batches = np.array_split(train_set, len(train_set[0]) // batch_size, axis=1)

    print(time.time() - t)
    print("train set prepared.")
    return batches


def get_pre_train_state(neuMF_state, gmf_state, mlp_state, alpha=0.5):
    '''
    updates neuMF_state's weights by gmf_state and mlp_state.
    :param alpha: indicates donation of gmf and mlp in last layer of neuMF.
    :return: transformed neuMF_state
    '''
    # Set matrix factorization embeddings.
    neuMF_state['user_matrix_factorization_embedding.weight'] = \
        gmf_state['user_matrix_factorization_embedding.weight']
    neuMF_state['item_matrix_factorization_embedding.weight'] = \
        gmf_state['item_matrix_factorization_embedding.weight']

    # Set multilayer perceptron embeddings.
    neuMF_state['user_multilayer_perceptron_embedding.weight'] = \
        mlp_state['user_multilayer_perceptron_embedding.weight']
    neuMF_state['item_multilayer_perceptron_embedding.weight'] = \
        mlp_state['item_multilayer_perceptron_embedding.weight']

    # Set multilayer perceptron fully connected weights and biases.
    for i in range(2):
        # Look at model definitions for naming policy. (2 * i)
        prefix = 'mlp_fully_connected_layers' + '.' + str(2 * i)  # eg. mlp_fully_connected_layers.2
        for postfix in ['weight', 'bias']:
            key = prefix + '.' + postfix  # eg. mlp_fully_connected_layers.2.weights
            neuMF_state[key] = mlp_state[key]

    # Set last layer of neuMF which is combining of last layer of mlp and gmf.
    # Weights.
    neuMF_state['neuMF_layer.0.weight'] = torch.cat([gmf_state['gmf_layer.0.weight'] * alpha,
                                                     mlp_state['mlp_layer.0.weight'] * (1 - alpha)],
                                                    dim=-1)
    # Biases. (each linear have one bias so we can not concat them, set wighted sum of them.)
    neuMF_state['neuMF_layer.0.bias'] = \
        (gmf_state['gmf_layer.0.bias'] * alpha).add(mlp_state['mlp_layer.0.bias'] * (1 - alpha))

    return neuMF_state
