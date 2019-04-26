import torch
from torch import nn


class NeuMF(nn.Module):
    def forward(self, user_indices, item_indices):
        # Multilayer perceptron embeddings.
        mlp_user_embeddings = self.user_multilayer_perceptron_embedding(user_indices)
        mlp_item_embeddings = self.item_multilayer_perceptron_embedding(item_indices)

        # Matrix factorization embeddings.
        mf_user_embeddings = self.user_matrix_factorization_embedding(user_indices)
        mf_item_embeddings = self.item_matrix_factorization_embedding(item_indices)

        # Concatenating mlp user and item embeddings to get input of mlp layers.
        mlp_input = torch.cat([mlp_user_embeddings, mlp_item_embeddings], dim=-1)

        # Pair-wise product of matrix factorization embeddings to get half of neuMF layer.
        mf_output = torch.mul(mf_user_embeddings, mf_item_embeddings)

        mlp_output = self.mlp_fully_connected_layers(mlp_input)

        neuMF_input = torch.cat([mf_output, mlp_output], dim=-1)

        output = self.neuMF_layer(neuMF_input)

        return output

    def __init__(self, num_users, num_items, latent_dim_for_mf, mlp_layer_sizes):
        super(NeuMF, self).__init__()

        # Matrix factorization embedding section.
        self.user_matrix_factorization_embedding = nn.Embedding(num_embeddings=num_users,
                                                                embedding_dim=latent_dim_for_mf)
        self.item_matrix_factorization_embedding = nn.Embedding(num_embeddings=num_items,
                                                                embedding_dim=latent_dim_for_mf)

        # Multilayer perceptron embedding section.
        mlp_embedding_dim = mlp_layer_sizes[0] // 2
        self.user_multilayer_perceptron_embedding = nn.Embedding(num_embeddings=num_users,
                                                                 embedding_dim=mlp_embedding_dim)
        self.item_multilayer_perceptron_embedding = nn.Embedding(num_embeddings=num_items,
                                                                 embedding_dim=mlp_embedding_dim)

        # Multilayer perceptron layers.
        self.mlp_fully_connected_layers = nn.Sequential()
        for i in range(len(mlp_layer_sizes) - 1):
            self.mlp_fully_connected_layers.add_module(
                # Linear layers have id 0, 2, 4, ...
                name=str(2 * i),
                module=nn.Linear(in_features=mlp_layer_sizes[i], out_features=mlp_layer_sizes[i+1]))

            self.mlp_fully_connected_layers.add_module(name=str(2 * i + 1), module=nn.ReLU())

        # Last layer of model which combines multilayer perceptron and matrix factorization parts.
        self.neuMF_layer = nn.Sequential(
            nn.Linear(in_features=mlp_layer_sizes[-1] + latent_dim_for_mf, out_features=1),
            nn.Sigmoid()
        )


class GMF(nn.Module):
    def forward(self, user_indices, item_indices):
        user_embeddings = self.user_matrix_factorization_embedding(user_indices)
        item_embeddings = self.item_matrix_factorization_embedding(item_indices)

        # Element-wise Product of user and item embeddings.
        matrix_factorization_vector = torch.mul(user_embeddings, item_embeddings)

        output = self.gmf_layer(matrix_factorization_vector)

        return output

    def __init__(self, num_users, num_items, latent_dimension):
        super(GMF, self).__init__()

        # Embeddings.
        self.user_matrix_factorization_embedding = nn.Embedding(num_embeddings=num_users,
                                                                embedding_dim=latent_dimension)
        self.item_matrix_factorization_embedding = nn.Embedding(num_embeddings=num_items,
                                                                embedding_dim=latent_dimension)

        self.gmf_layer = nn.Sequential(
            nn.Linear(latent_dimension, 1),
            nn.Sigmoid()
        )


class MLP(nn.Module):
    def forward(self, user_indices, item_indices):
        user_embeddings = self.user_multilayer_perceptron_embedding(user_indices)
        item_embeddings = self.item_multilayer_perceptron_embedding(item_indices)

        # Concatenation of user and item embeddings to feed into first layer of mlp.
        input = torch.cat([user_embeddings, item_embeddings], dim=-1)

        mlp_layer_input = self.mlp_fully_connected_layers(input)

        output = self.mlp_layer(mlp_layer_input)

        return output

    def __init__(self, num_users, num_items, layer_sizes):
        super(MLP, self).__init__()

        # Embeddings.
        mlp_embedding_dim = layer_sizes[0] // 2
        self.user_multilayer_perceptron_embedding = nn.Embedding(num_embeddings=num_users,
                                                                 embedding_dim=mlp_embedding_dim)
        self.item_multilayer_perceptron_embedding = nn.Embedding(num_embeddings=num_items,
                                                                 embedding_dim=mlp_embedding_dim)

        # Multilayer perceptron layers.
        self.mlp_fully_connected_layers = nn.Sequential()
        for i in range(len(layer_sizes) - 1):
            self.mlp_fully_connected_layers.add_module(
                # Linear layers have id 0, 2, 4, ...
                name=str(2 * i),
                module=nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
            )

            self.mlp_fully_connected_layers.add_module(name=str(2 * i + 1), module=nn.ReLU())

        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=layer_sizes[-1], out_features=1),
            nn.Sigmoid()
        )
