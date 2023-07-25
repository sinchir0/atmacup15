import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(33)


class NCF(nn.Module):
    def __init__(
        self, num_users, num_items, emb_dim=100, hidden_layers=[64, 32, 16], dropout=0.1
    ):
        super().__init__()

        # MF
        self.mf_user_vector = nn.Embedding(num_users, emb_dim)
        self.mf_item_vector = nn.Embedding(num_items, emb_dim)
        self.mlp_user_vector = nn.Embedding(num_users, emb_dim)
        self.mlp_item_vector = nn.Embedding(num_items, emb_dim)

        # Define MLP layers
        mlp_layers = []
        input_dim = emb_dim * 2
        for hidden_dim in hidden_layers:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        mlp_layers.append(nn.Linear(hidden_layers[-1], emb_dim))
        self.mlp_layer = nn.Sequential(*mlp_layers)

        self.neufm_layer = nn.Sequential(nn.Linear(emb_dim * 2, 1), nn.Sigmoid())

    def forward(self, user_indices, item_indices):
        mf_user_vector = self.mf_user_vector(user_indices)
        mlp_user_vector = self.mlp_user_vector(user_indices)

        mf_item_vector = self.mf_item_vector(item_indices)
        mlp_item_vector = self.mlp_item_vector(item_indices)

        gmf_layer_output = torch.mul(mf_user_vector, mf_item_vector)
        mlp_output = self.mlp_layer(
            torch.cat([mlp_user_vector, mlp_item_vector], dim=-1)
        )

        output = self.neufm_layer(torch.cat([gmf_layer_output, mlp_output], dim=-1))

        return output.squeeze()


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


def root_mean_squared_error(y_true, y_pred):
    """mean_squared_error の root (0.5乗)"""
    return mean_squared_error(y_true, y_pred) ** 0.5


if __name__ == "__main__":

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
    test_data["user_id"] = test_data["user_id"].apply(
        lambda x: x if x in user_id_encoder.classes_ else "unknown_user"
    )
    test_data["anime_id"] = test_data["anime_id"].apply(
        lambda x: x if x in anime_id_encoder.classes_ else "unknown_anime"
    )

    # Initialize a scaler
    scaler = MinMaxScaler()

    # scoreを正規化する
    train_data["org_score"] = train_data["score"]
    train_data["score"] = scaler.fit_transform(train_data["score"].values.reshape(-1, 1))

    # encoderのunknown_user, unknown_animeを追加する
    user_id_encoder.classes_ = np.concatenate((user_id_encoder.classes_, ["unknown_user"]))
    anime_id_encoder.classes_ = np.concatenate(
        (anime_id_encoder.classes_, ["unknown_anime"])
    )

    # testデータのencode
    test_data["user_id"] = user_id_encoder.transform(test_data["user_id"])
    test_data["anime_id"] = anime_id_encoder.transform(test_data["anime_id"])

    # train, validに分ける
    train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Datasetの作成
    train_dataset = AnimeDataset(
        train_data["user_id"].values,
        train_data["anime_id"].values,
        train_data["score"].values,
    )
    valid_dataset = AnimeDataset(
        valid_data["user_id"].values,
        valid_data["anime_id"].values,
        valid_data["score"].values,
    )
    test_dataset = AnimeDataset(test_data["user_id"].values, test_data["anime_id"].values)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # モデルの初期化
    num_users = len(user_id_encoder.classes_)
    num_items = len(anime_id_encoder.classes_)
    model = NCF(num_users, num_items)

    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=0.001)
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 学習
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
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
                # 0~1で出力されたscoreを1~10に戻す
                outputs = scaler.inverse_transform(outputs.detach().numpy().reshape(-1, 1))
                valid_pred.extend(outputs)
        valid_score = root_mean_squared_error(valid_data["org_score"].values, valid_pred)
        print(f"Valid RMSE: {valid_score}")
        if best_score > valid_score:
            print("Best Score")
            best_score = valid_score
            torch.save(model.state_dict(), "ipynb/NCF/ncf_anime_feat_best_model.pth")

    # 推論
    model.eval()

    predictions = np.array([])

    # load best model
    model.load_state_dict(torch.load("ipynb/NCF/ncf_anime_feat_best_model.pth"))

    with torch.no_grad():
        for user_ids, anime_ids in test_dataloader:
            user_ids = user_ids.to(device)
            anime_ids = anime_ids.to(device)

            batch_predictions = model(user_ids, anime_ids)
            batch_predictions = scaler.inverse_transform(
                batch_predictions.detach().numpy().reshape(-1, 1)
            )

            predictions = np.append(predictions, batch_predictions)

    # subできる形式に変更する
    sample_submission["score"] = predictions
    sample_submission.to_csv("ipynb/NCF/ncf_gpt.csv", index=False)
