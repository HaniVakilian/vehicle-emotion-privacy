import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load specific client data
df = pd.read_csv("data/client_1_data_V2.csv")  # 👈 Change to client_2 or client_3 in other files

# Encode emotion labels
df["Emotion_Mapped"] = LabelEncoder().fit_transform(df["Emotion"])

# Prepare data
X = df.drop(columns=["Emotion", "Emotion_Mapped"]).values.astype("float32")
y = df["Emotion_Mapped"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the model
NUM_CLASSES = 6  # Always use 6, even if some classes are missing in the client's data
model = nn.Sequential(
    nn.Linear(X.shape[1], 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, NUM_CLASSES)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    outputs = model(X_test_tensor)
    loss = loss_fn(outputs, y_test_tensor).item()
    accuracy = (outputs.argmax(1) == y_test_tensor).float().mean().item()
    return loss, accuracy

def get_model_parameters():
    return [val.cpu().numpy() for val in model.state_dict().values()]

def set_model_parameters(parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_model_parameters()

    def fit(self, parameters, config):
        try:
            set_model_parameters(parameters)
            train()
            return get_model_parameters(), len(X_train_tensor), {}
        except Exception as e:
            print(f"[ERROR] Fit failed: {e}")
            raise

    def evaluate(self, parameters, config):
        set_model_parameters(parameters)
        loss, accuracy = test()
        return float(loss), len(X_test_tensor), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient())
