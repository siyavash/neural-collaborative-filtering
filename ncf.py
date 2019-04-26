import math
import time
from random import randrange

import numpy as np
import torch as torch
from torch import optim, nn, tensor, Tensor
from torch.autograd import Variable

import util
from models import GMF, MLP, NeuMF


def evaluate(model, data, test_set, top_k, num_others):
    '''
    Evaluate given model and report hit ratio and normalized discounted cumulative gain.
    Add num_others non interacted user-item per test.
    :param num_others: number of non interacted user-items per test.
    '''
    users, items = [], []
    for user, item in enumerate(test_set[1:], start=1):
        users.append(user)
        items.append(item)
        # Sampling num_others random non interacted user-items.
        for i in range(num_others):
            rand_item = randrange(1, num_items)
            while rand_item in data[user]:
                rand_item = randrange(1, num_items)

            users.append(user)
            items.append(rand_item)

    user_indices = Variable(tensor(np.asarray(users, dtype=int)))
    item_indices = Variable(tensor(np.asarray(items, dtype=int)))

    result = model(user_indices, item_indices)
    # Converting to numpy array.
    result = result.detach().numpy()
    # Reshaping to 1d array. (x, 1) to (x)
    result = result.reshape((result.shape[0]))
    # Splitting result to get each user's result.
    user_results = np.array_split(result, result.shape[0] // (num_others + 1))
    hit_sum = 0
    ndcg_sum = 0
    for user_result in user_results:
        item_rank = (user_result > user_result[0]).sum()
        if item_rank < top_k:
            hit_sum += 1
            ndcg_sum += math.log(2) / math.log(2 + item_rank)

    num_users = len(data.keys())
    print("hit ratio average: ", hit_sum / num_users)
    print("ncdg average: ", ndcg_sum / num_users)


def train(model, data, train_set, test_set, loss_function, optimizer, save_dir,
          top_k=10, num_others=100, num_epochs=10, evaluate_per_k_epochs=2, batch_size=256):
    '''
    Training model by given parameters and evaluating per evaluate_per_k_epochs epoch.
    '''
    evaluate(model, data, test_set, top_k, num_others)

    for epoch in range(1, num_epochs + 1):
        print("starting epoch {}.".format(epoch))
        train_set = util.get_train_set(data, test_set, num_items,
                                       num_negative_samples=4, batch_size=batch_size)
        t = time.time()
        for batch in train_set:
            user_indices = Variable(tensor(batch[0]))
            item_indices = Variable(tensor(batch[1]))
            labels = Variable(Tensor(batch[2].reshape(-1, 1)))

            result = model(user_indices, item_indices)
            # Calculate loss.
            loss = loss_function(result, labels)

            # Update grads and different of weight change per edge.
            optimizer.zero_grad()
            loss.backward()

            # Update weights.
            optimizer.step()

        print("epoch {} completed in {} secs.".format(epoch, time.time() - t))
        if epoch % evaluate_per_k_epochs == 0:
            evaluate(model, data, test_set, top_k, num_others)

    print("Saving model to ", save_dir)
    torch.save(model.state_dict(), save_dir)
    print("Model saved.")

    evaluate(model, data, test_set, top_k, num_others)

# parameters
dataset = "ml-1m"
delimiter = '::'
data_name = "ratings.dat"
num_negative_samples = 4
batch_size = 256
mf_latent_size = 8
learning_rate = 0.005
mlp_layer_sizes = [64, 32, 16, 8]
evaluation_top_k = 10

# Create dictionary of data.
data, num_users, num_items = util.get_data(dataset + '/' + data_name, delimiter)

test_set = util.get_test_set(data, num_users)
train_set = util.get_train_set(data, test_set, num_items, num_negative_samples, batch_size)

# Create general matrix factorization model.
gmf_model = GMF(num_users, num_items, latent_dimension=mf_latent_size)
print(gmf_model)
gmf_model.apply(util.init_weights)

train(model=gmf_model,
      data=data,
      train_set=train_set,
      test_set=test_set,
      loss_function=nn.BCELoss(),
      optimizer=optim.Adam(gmf_model.parameters(), lr=learning_rate),
      save_dir=dataset + '/gmf',
      top_k=evaluation_top_k)

# Create multilayer percetpron model.
mlp_model = MLP(num_users, num_items, mlp_layer_sizes)
print(mlp_model)
mlp_model.apply(util.init_weights)

train(model=mlp_model,
      data=data,
      train_set=train_set,
      test_set=test_set,
      loss_function=nn.BCELoss(),
      optimizer=optim.Adam(mlp_model.parameters(), lr=learning_rate),
      save_dir=dataset + '/mlp',
      top_k=evaluation_top_k)

neuMF_model = NeuMF(num_users, num_items, mf_latent_size, mlp_layer_sizes)
print(neuMF_model)

# Load gmf and mlp models.
gmf_state_dict = torch.load(dataset + '/gmf')
mlp_state_dict = torch.load(dataset + '/mlp')
neuMF_stat = neuMF_model.state_dict()

neuMF_stat = util.get_pre_train_state(neuMF_stat, gmf_state_dict, mlp_state_dict)
# Set weights of neuMF by gmf and mlp.
neuMF_model.load_state_dict(neuMF_stat)

train(model=neuMF_model,
      data=data,
      train_set=train_set,
      test_set=test_set,
      loss_function=nn.BCELoss(),
      optimizer=optim.SGD(neuMF_model.parameters(), lr=learning_rate),
      save_dir=dataset + '/neumf',
      top_k=evaluation_top_k)
