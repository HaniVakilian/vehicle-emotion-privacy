# server.py
import os
import flwr as fl
from flwr.server.strategy import FedAvg

os.makedirs("results", exist_ok=True)

class LoggingFedAvg(FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        # Call FedAvg's default aggregation first (gets global loss, metrics dict)
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        # --- Robust weighted accuracy over client evals (works across Flower versions)
        # results: List[Tuple[ClientProxy, EvaluateRes]]
        # EvaluateRes has: .loss, .num_examples, .metrics (dict)
        num_total = 0
        acc_weighted_sum = 0.0
        for _, res in results:
            if res is None:
                continue
            n = getattr(res, "num_examples", 0) or 0
            m = getattr(res, "metrics", {}) or {}
            acc = m.get("accuracy", None)
            if acc is not None and n > 0:
                acc_weighted_sum += float(acc) * int(n)
                num_total += int(n)

        acc_global = (acc_weighted_sum / num_total) if num_total > 0 else None

        # Also get the aggregated loss from FedAvg (if available)
        loss_global = None
        if aggregated is not None:
            loss_global, _metrics_global = aggregated

        # Log to CSV
        with open("results/server_metrics.csv", "a") as f:
            f.write(
                f"{server_round},{'' if loss_global is None else loss_global},"
                f"{'' if acc_global is None else acc_global}\n"
            )

        return aggregated

if __name__ == "__main__":
    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,          # 3 training clients
        min_evaluate_clients=4,     # 3 train + 1 test client
        min_available_clients=4,    # wait for all 4 to connect
        accept_failures=False,
        # NOTE: no evaluate_metrics_aggregation_fn here (we aggregate above)
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )
