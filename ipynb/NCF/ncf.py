import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class NCF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim, layers, num_anime_features):
        super(NCF, self).__init__()

        # MF layers
        self.user_embed_MF = nn.Embedding(num_users, mf_dim)
        self.item_embed_MF = nn.Embedding(num_items, mf_dim)

        # MLP layers
        self.user_embed_MLP = nn.Embedding(num_users, layers[0] // 2)
        self.item_embed_MLP = nn.Embedding(num_items, layers[0] // 2)
        
        # Anime features embedding layer
        self.anime_feature_embed = nn.Linear(num_anime_features, layers[0] // 2)
        
        # self.mlp = nn.Sequential([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.mlp = nn.Sequential(
            nn.Linear(3 * (layers[0] // 2), 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        )

        # final prediction layer
        # self.prediction = nn.Linear(mf_dim + layers[-1], 1)
        self.prediction = nn.Linear(40, 1)

    def forward(self, user_indices, item_indices, anime_features):
        # MF part
        user_embed_MF = self.user_embed_MF(user_indices)
        item_embed_MF = self.item_embed_MF(item_indices)
        mf_vector = user_embed_MF * item_embed_MF

        # MLP part
        user_embed_MLP = self.user_embed_MLP(user_indices)
        item_embed_MLP = self.item_embed_MLP(item_indices)
        
        # Anime features part
        anime_feature_embed = self.anime_feature_embed(anime_features)
        
        # user_embed_MLP (layers[0] // 2)
        # item_embed_MLP (layers[0] // 2)
        # anime_feature_embed (layers[0] // 2)

        mlp_vector = torch.cat([user_embed_MLP, item_embed_MLP, anime_feature_embed], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        # concatenate MF and MLP parts
        vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        # final prediction
        pred = self.prediction(vector)
        return torch.sigmoid(pred)

class RatingDataset(Dataset):
    """Rating Dataset for DataLoader"""

    def __init__(self, user_tensor, item_tensor, feature_tensor, target_tensor=None):
        """
        Args:
            user_tensor (torch.Tensor): User ID tensor. Size [n_samples]
            item_tensor (torch.Tensor): Item ID tensor. Size [n_samples]
            feature_tensor (torch.Tensor): Feature tensor. Size [n_samples, n_features]
            target_tensor (torch.Tensor, optional): Target tensor. Size [n_samples]
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.feature_tensor = feature_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        if self.target_tensor is None:
            return self.user_tensor[index], self.item_tensor[index], self.feature_tensor[index]
        else:
            return self.user_tensor[index], self.item_tensor[index], self.feature_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)
    
    # Reload the data
train_data = pd.read_csv('data/train.csv')
anime_data = pd.read_csv('data/anime.csv')
test_data = pd.read_csv('data/test.csv')

print(train_data.shape)
print(anime_data.shape)
print(test_data.shape)

# Create anime_features
anime_features = anime_data[['anime_id', 'episodes', 'members', 'watching', 'completed', 'on_hold', 'dropped', 'plan_to_watch']]

# Merge train data and anime features
train_data = train_data.merge(anime_features, on='anime_id', how='left')

# Merge test data and anime features
test_data = test_data.merge(anime_features, on='anime_id', how='left')

# Get unique user and item IDs
unique_user_ids = pd.unique(pd.concat([train_data['user_id'], test_data['user_id']]))
unique_item_ids = pd.unique(pd.concat([train_data['anime_id'], test_data['anime_id']]))

# Create user and item ID to index mapping
user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
item_id_to_index = {item_id: index for index, item_id in enumerate(unique_item_ids)}

# Apply mapping to train and test data
train_data['user_id'] = train_data['user_id'].map(user_id_to_index)
train_data['anime_id'] = train_data['anime_id'].map(item_id_to_index)
test_data['user_id'] = test_data['user_id'].map(user_id_to_index)
test_data['anime_id'] = test_data['anime_id'].map(item_id_to_index)

# Replace 'Unknown' with NaN in 'episodes' column
train_data['episodes'] = train_data['episodes'].replace('Unknown', np.nan)
test_data['episodes'] = test_data['episodes'].replace('Unknown', np.nan)

# Convert 'episodes' column to float type
train_data['episodes'] = train_data['episodes'].astype(float)
test_data['episodes'] = test_data['episodes'].astype(float)

# Fill NaN with the median of the column
train_data['episodes'].fillna(train_data['episodes'].median(), inplace=True)
test_data['episodes'].fillna(test_data['episodes'].median(), inplace=True)

# Prepare training data
train_user_tensor = torch.from_numpy(train_data['user_id'].values.astype(np.int64))
train_item_tensor = torch.from_numpy(train_data['anime_id'].values.astype(np.int64))
train_feature_tensor = torch.from_numpy(train_data[anime_features.columns].values.astype(np.float32))
train_target_tensor = torch.from_numpy(train_data['score'].values.astype(np.float32))
train_dataset = RatingDataset(train_user_tensor, train_item_tensor, train_feature_tensor, train_target_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Prepare test data
test_user_tensor = torch.from_numpy(test_data['user_id'].values.astype(np.int64))
test_item_tensor = torch.from_numpy(test_data['anime_id'].values.astype(np.int64))
test_feature_tensor = torch.from_numpy(test_data[anime_features.columns].values.astype(np.float32))
test_dataset = RatingDataset(test_user_tensor, test_item_tensor, test_feature_tensor, target_tensor=None)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize model
num_users = len(train_data['user_id'].unique())
num_items = len(train_data['anime_id'].unique())
num_anime_features = len(anime_features.columns)
mf_dim = 8  # dimension of MF
layers = [384, 128, 32, 8]  # layer size of MLP
model = NCF(num_users, num_items, mf_dim, layers, num_anime_features)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in tqdm(range(1)):  # run for 10 epochs
    for user, item, feature, rating in train_loader:
        # Forward pass
        outputs = model(user, item, feature)
        loss = criterion(outputs.squeeze(), rating)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{10}, Loss: {loss.item()}')

# Make predictions on test set
model.eval()  # switch to evaluation mode
with torch.no_grad():
    predictions = []
    for user, item, feature in test_loader:
        outputs = model(user, item, feature)
        predictions.extend(outputs.squeeze().tolist())

# Print first 10 predictions
print('First 10 predictions:', predictions[:10])