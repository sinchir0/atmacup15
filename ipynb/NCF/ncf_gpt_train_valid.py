import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error


class AnimeDataset(Dataset):
    def __init__(self, user_ids, anime_ids, scores=None):
        self.user_ids = user_ids
        self.anime_ids = anime_ids
        self.scores = scores

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.scores is None:
            return self.user_ids[idx], self.anime_ids[idx]
        return self.user_ids[idx], self.anime_ids[idx], self.scores[idx]

class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=100, hidden_layers=[64, 32, 16], dropout=0.1):
        super().__init__()

        # MF part
        self.user_embed_MF = nn.Embedding(num_users, emb_dim)
        self.item_embed_MF = nn.Embedding(num_items, emb_dim)

        # MLP layers
        self.mlp = nn.Sequential(*[
            nn.Linear(emb_dim * 2, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            *sum([[nn.Linear(hidden_layers[i], hidden_layers[i+1]), nn.ReLU(), nn.Dropout(dropout)] for i in range(len(hidden_layers) - 1)], []),
            nn.Linear(hidden_layers[-1], emb_dim)
        ])
        print(self.mlp)
        self.final_layer = nn.Sequential(
            nn.Linear(emb_dim * 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embed_MF(user_indices)
        item_emb = self.item_embed_MF(item_indices)
        mf_output = user_emb * item_emb
        mlp_output = self.mlp(torch.cat([user_emb, item_emb], dim=-1))
        final_output = self.final_layer(torch.cat([mf_output, mlp_output], dim=-1))
        return final_output.squeeze()

# Load the data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
sample_submission = pd.read_csv("data/sample_submission.csv")

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
train_dataset = AnimeDataset(train_data["user_id"].values, train_data["anime_id"].values, train_data["score"].values)
valid_dataset = AnimeDataset(valid_data["user_id"].values, valid_data["anime_id"].values, valid_data["score"].values)
test_dataset = AnimeDataset(test_data["user_id"].values, test_data["anime_id"].values)

# Create the PyTorch dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
num_users = len(user_id_encoder.classes_)
num_items = len(anime_id_encoder.classes_)
model = NCF(num_users, num_items)

# Define the loss function
# criterion = nn.BCELoss()
criterion = nn.MSELoss()

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Define the number of training epochs
epochs = 10

# Move the model to the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def root_mean_squared_error(y_true, y_pred):
    """mean_squared_error の root (0.5乗)"""
    return mean_squared_error(y_true, y_pred) ** .5

# Train the model for one epoch
best_score = np.inf
for epoch in range(epochs):
    for i, (user_ids, anime_ids, scores) in enumerate(train_dataloader):
        # Move the data to the correct device
        user_ids = user_ids.to(device)
        anime_ids = anime_ids.to(device)
        scores = scores.float().to(device)

        # Forward pass
        outputs = model(user_ids, anime_ids)

        # Compute the loss
        loss = criterion(outputs, scores)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    # calc valid score
    valid_pred = []
    with torch.no_grad():
        for user_ids, anime_ids, scores in valid_dataloader:
            # Move the data to the correct device
            user_ids = user_ids.to(device)
            anime_ids = anime_ids.to(device)
            scores = scores.float().to(device)
            # Forward pass
            outputs = model(user_ids, anime_ids)
            valid_pred.extend(outputs.cpu().numpy())
    valid_score = root_mean_squared_error(valid_data["score"].values, valid_pred)
    print(f"Valid RMSE: {valid_score}")
    if best_score > valid_score:
        print("Best Score")
        best_score = valid_score
        torch.save(model.state_dict(), "ipynb/NCF/ncf_anime_feat_best_model.pth")

# Put the model in evaluation mode
model.eval()

# Initialize an empty numpy array to store the predictions
predictions = np.array([])

# Iterate over the test data and make predictions
with torch.no_grad():
    for user_ids, anime_ids in test_dataloader:
        # Move the data to the correct device
        user_ids = user_ids.to(device)
        anime_ids = anime_ids.to(device)

        # Make predictions
        batch_predictions = model(user_ids, anime_ids)

        # Store the predictions
        predictions = np.append(predictions, batch_predictions.cpu().numpy())

# subできる形式に変更する
sample_submission["score"] = predictions
sample_submission.to_csv("ncf_gpt.csv", index=False)