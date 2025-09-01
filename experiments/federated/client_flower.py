# client_flower.py
import os, argparse, random, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import flwr as fl

# ---- determinism
os.environ["PYTHONHASHSEED"] = "0"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- constants
NUM_CLASSES = 6
BATCH_SIZE = 32
LOCAL_EPOCHS = 10
LR = 5e-4

# ---- fixed label map
EMOTION_MAP = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "fear": 3,
    "surprise": 4,
    "disgust": 5,
}

# ---- load dataset
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # ensure label column is clean
    df["Emotion"] = df["Emotion"].astype(str).str.lower().str.strip()
    if not set(df["Emotion"].unique()).issubset(EMOTION_MAP.keys()):
        raise ValueError(f"Unexpected emotions found: {set(df['Emotion'].unique())}")

    y = df["Emotion"].map(EMOTION_MAP).astype("int64").values

    # fixed feature columns
    X = df[["Heart_Rate_(bpm)", "Skin_Conductance"]].astype("float32").values

    # split train/test
    strat = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    # tensors
    Xtr, ytr = torch.tensor(X_train), torch.tensor(y_train)
    Xte, yte = torch.tensor(X_test),  torch.tensor(y_test)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, len(Xtr), len(Xte)

# ---- model
def build_model():
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, NUM_CLASSES),
    )

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        tot_loss += loss.item() * yb.size(0)
        tot_correct += (logits.argmax(1) == yb).sum().item()
        tot += yb.size(0)
    return (tot_loss / max(tot, 1)), (tot_correct / max(tot, 1))

def get_params(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_params(model, params):
    sd = model.state_dict()
    new_sd = {k: torch.tensor(v) for k, v in zip(sd.keys(), params)}
    model.load_state_dict(new_sd, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_path, out_path):
        self.train_loader, self.test_loader, self.ntr, self.nte = load_data(data_path)
        self.model = build_model()
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()
        total, n = 0.0, 0
        for _ in range(LOCAL_EPOCHS):
            for xb, yb in self.train_loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total += loss.item() * yb.size(0)
                n += yb.size(0)
        avg_train_loss = total / max(n, 1)
        return get_params(self.model), self.ntr, {"train_loss": float(avg_train_loss)}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader)
        rnd = int(config.get("server_round", 0))
        with open(self.out_path, "a") as f:
            f.write(f"{rnd},{loss:.6f},{acc:.6f}\n")
        return float(loss), self.nte, {"accuracy": float(acc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to client CSV")
    ap.add_argument("--out", required=True, help="Results CSV path")
    args = ap.parse_args()

    client = FlowerClient(args.data, args.out).to_client()
    fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
