# Import necessary libraries
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# # Define the AnimeDataset
# class AnimeDataset(Dataset):
#     def __init__(self, user_ids, anime_ids, scores=None):
#         self.user_ids = user_ids
#         self.anime_ids = anime_ids
#         self.scores = scores

#     def __len__(self):
#         return len(self.user_ids)

#     def __getitem__(self, idx):
#         if self.scores is None:
#             return self.user_ids[idx], self.anime_ids[idx]
#         return self.user_ids[idx], self.anime_ids[idx], self.scores[idx]

class AnimeDataset(Dataset):
    """Rating Dataset for DataLoader"""

    def __init__(self, user_ids, anime_ids, anime_feature, score=None):
        self.user_ids = user_ids
        self.anime_ids = anime_ids
        self.anime_feature = anime_feature
        self.score = score

    def __getitem__(self, index):
        if self.score is None:
            return self.user_ids[index], self.anime_ids[index], self.anime_feature[index]
        else:
            return self.user_ids[index], self.anime_ids[index], self.anime_feature[index], self.score[index]

    def __len__(self):
        return len(self.user_ids)

# Define the NCF model
class NCF(nn.Module):
    # def __init__(self, num_users, num_items, num_anime_features, emb_dim=100, hidden_layers=[128, 384, 128, 64, 32], dropout=0.1):
    def __init__(self, num_users, num_items, num_anime_features, emb_dim=100, hidden_layers=[64, 32, 16], dropout=0.1):
        super().__init__()

        # MF part
        self.user_embed_MF = nn.Embedding(num_users, emb_dim)
        self.item_embed_MF = nn.Embedding(num_items, emb_dim)

        # MLP layers
        self.user_embed_MLP = nn.Embedding(num_users, hidden_layers[0] // 2)
        self.item_embed_MLP = nn.Embedding(num_items, hidden_layers[0] // 2)

        # Anime features embedding layer
        self.anime_feature_embed = nn.Linear(num_anime_features, hidden_layers[0] // 2)

        # MLP layers
        self.mlp = nn.Sequential(*[
            nn.Linear((hidden_layers[0] // 2) * 3, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            *sum([[nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU(), nn.Dropout(dropout)] for i in range(len(hidden_layers) - 1)], []),
            nn.Linear(hidden_layers[-1], emb_dim)
        ])
        print(self.mlp)
        self.final_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, 1),
        )

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

        mlp_vector = torch.cat([user_embed_MLP, item_embed_MLP, anime_feature_embed], dim=-1)
        mlp_vector = self.mlp(mlp_vector)

        # concatenate MF and MLP parts
        vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        final_output = self.final_layer(vector)
        return final_output.squeeze()

# Load the data
train_data = pd.read_csv("data/train.csv")
anime_data = pd.read_csv("data/anime.csv")
test_data = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

# Create anime_features
# anime_features_continuous_col = ['episodes', 'members', 'watching', 'completed', 'on_hold', 'dropped', 'plan_to_watch']
anime_features_continuous_col = ['dropped']
# anime_features = anime_data[['anime_id', 'episodes', 'members', 'watching', 'completed', 'on_hold', 'dropped', 'plan_to_watch']]
anime_features = anime_data[['anime_id', 'dropped']]

# episodesの欠損値は、medianで補完
# anime_features['episodes'] = anime_features['episodes'].replace('Unknown', np.nan)
# anime_features['episodes'] = anime_features['episodes'].astype(float)
# anime_features['episodes'] = anime_features['episodes'].fillna(anime_features['episodes'].median())

from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform the training data
anime_features[anime_features_continuous_col] = scaler.fit_transform(
    anime_features[anime_features_continuous_col]
)

# Merge anime data
train_data = train_data.merge(anime_features, on='anime_id', how='left')
test_data = test_data.merge(anime_features, on='anime_id', how='left')

# Create user and anime encoders
user_id_encoder = LabelEncoder()
anime_id_encoder = LabelEncoder()

unique_user_id = list(set(train_data["user_id"].tolist()))
unique_anime_id = list(set(train_data["anime_id"].tolist()))

user_id_encoder.fit(unique_user_id)
anime_id_encoder.fit(unique_anime_id)

train_data["user_id"] = user_id_encoder.transform(train_data["user_id"])
train_data["anime_id"] = anime_id_encoder.transform(train_data["anime_id"])

# trainに登場しないIDは全てUnknownにする
test_data["user_id"] = test_data["user_id"].apply(lambda x: x if x in user_id_encoder.classes_ else "unknown_user")
test_data["anime_id"] = test_data["anime_id"].apply(lambda x: x if x in anime_id_encoder.classes_ else "unknown_anime")

# Add the 'unknown' classes to the encoders
user_id_encoder.classes_ = np.concatenate((user_id_encoder.classes_, ['unknown_user']))
anime_id_encoder.classes_ = np.concatenate((anime_id_encoder.classes_, ['unknown_anime']))

test_data["user_id"] = user_id_encoder.transform(test_data["user_id"])
test_data["anime_id"] = anime_id_encoder.transform(test_data["anime_id"])

# split train to train, valid by train_test_split
from sklearn.model_selection import train_test_split

train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Create the PyTorch datasets
train_dataset = AnimeDataset(
    train_data["user_id"].values,
    train_data["anime_id"].values,
    train_data[anime_features.columns].values.astype(np.float32),
    train_data["score"].values
)

valid_dataset = AnimeDataset(
    valid_data["user_id"].values,
    valid_data["anime_id"].values,
    valid_data[anime_features.columns].values.astype(np.float32),
    valid_data["score"].values
)

test_dataset = AnimeDataset(
    test_data["user_id"].values,
    test_data["anime_id"].values,
    test_data[anime_features.columns].values.astype(np.float32)
)

# Create the PyTorch dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
num_users = len(user_id_encoder.classes_)
num_items = len(anime_id_encoder.classes_)
num_anime_features = len(anime_features.columns)
model = NCF(num_users, num_items, num_anime_features)

# Define the loss function
# criterion = nn.BCELoss()
criterion = nn.MSELoss()

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=0.001)

# Define the number of training epochs
epochs = 20

# Move the model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def root_mean_squared_error(y_true, y_pred):
    """mean_squared_error の root (0.5乗)"""
    return mean_squared_error(y_true, y_pred) ** .5

# Train the model for one epoch
best_score = np.inf
for epoch in range(epochs):
    for user_ids, anime_ids, anime_feature, scores in train_dataloader:
        # Move the data to the correct device
        user_ids = user_ids.to(device)
        anime_ids = anime_ids.to(device)
        scores = scores.float().to(device)

        # Forward pass
        outputs = model(user_ids, anime_ids, anime_feature)
        
        # Compute the loss
        loss = criterion(outputs, scores)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calc valid score
    valid_pred = []
    with torch.no_grad():
        for user_ids, anime_ids, anime_feature, scores in valid_dataloader:
            # Move the data to the correct device
            user_ids = user_ids.to(device)
            anime_ids = anime_ids.to(device)
            scores = scores.float().to(device)
            # Forward pass
            outputs = model(user_ids, anime_ids, anime_feature)
            valid_pred.extend(outputs.cpu().numpy())
    valid_score = root_mean_squared_error(valid_data["score"].values, valid_pred)
    print(f"Valid RMSE: {valid_score}")
    if best_score > valid_score:
        print("Best Score")
        best_score = valid_score
        torch.save(model.state_dict(), "ipynb/NCF/ncf_anime_feat_best_model.pth")

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Put the model in evaluation mode
model.eval()

# Initialize an empty numpy array to store the predictions
predictions = np.array([])

# Iterate over the test data and make predictions
with torch.no_grad():
    for user_ids, anime_ids, anime_feature in test_dataloader:
        # Move the data to the correct device
        user_ids = user_ids.to(device)
        anime_ids = anime_ids.to(device)

        # Make predictions
        batch_predictions = model(user_ids, anime_ids, anime_feature)

        # Store the predictions
        predictions = np.append(predictions, batch_predictions.cpu().numpy())

# subできる形式に変更する
sample_submission["score"] = predictions
sample_submission.to_csv("ipynb/NCF/ncf_anime_feat.csv", index=False)