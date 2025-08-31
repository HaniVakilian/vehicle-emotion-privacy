import flwr as fl

def main():
    # Define strategy (with logging capabilities)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,
        min_available_clients=3,
        on_fit_config_fn=None,  # You can optionally send config to clients
        on_evaluate_config_fn=None,
    )

    # Start server and collect training history
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Optional: Save history to CSV
    if history and history.losses_distributed:
        with open("results/fl_metrics.csv", "w") as f:
            f.write("round,loss\n")
            for rnd, loss in enumerate(history.losses_distributed):
                f.write(f"{rnd+1},{loss}\n")

if __name__ == "__main__":
    main()
