# test_client.py
import os, argparse, random, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import flwr as fl

os.environ["PYTHONHASHSEED"]="0"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

NUM_CLASSES = 6
BATCH_SIZE = 64

EMOTION_MAP = {"happy":0,"sad":1,"angry":2,"fear":3,"surprise":4,"disgust":5}

def load_test(csv_path):
    df = pd.read_csv(csv_path)
    df["Emotion"] = df["Emotion"].astype(str).str.lower().str.strip()

    y = df["Emotion"].map(EMOTION_MAP).astype("int64").values
    X = df[["Heart_Rate_(bpm)", "Skin_Conductance"]].astype("float32").values

    # Scale for evaluation (fit scaler on the test data itself)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader, len(X)

def build_model():
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, NUM_CLASSES),
    )

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        tot_loss += loss.item() * yb.size(0)
        tot_correct += (logits.argmax(1) == yb).sum().item()
        tot += yb.size(0)
    return (tot_loss / max(tot,1)), (tot_correct / max(tot,1))

def get_params(model):
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_params(model, params):
    sd = model.state_dict()
    new_sd = {k: torch.tensor(v) for k, v in zip(sd.keys(), params)}
    model.load_state_dict(new_sd, strict=True)

class TestOnlyClient(fl.client.NumPyClient):
    def __init__(self, test_csv, out_path):
        self.test_loader, self.nte = load_test(test_csv)
        self.model = build_model()
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def get_parameters(self, config):
        # Provide initial params (unused for training here)
        return get_params(self.model)

    def fit(self, parameters, config):
        # Do NOT train; just pass weights through unchanged
        set_params(self.model, parameters)
        return get_params(self.model), 0, {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        loss, acc = eval_model(self.model, self.test_loader)
        rnd = int(config.get("server_round", 0))
        with open(self.out_path, "a") as f:
            f.write(f"{rnd},{loss:.6f},{acc:.6f}\n")
        return float(loss), self.nte, {"accuracy": float(acc)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to TEST CSV")
    ap.add_argument("--out", required=True, help="results path, e.g. results/test_metrics.csv")
    args = ap.parse_args()
    client = TestOnlyClient(args.data, args.out).to_client()
    fl.client.start_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
