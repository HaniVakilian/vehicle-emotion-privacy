import flwr as fl

def main():
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(
            min_fit_clients=3,       # require all 3 clients in each round
            min_available_clients=3  # server will wait for 3 clients
        )
    )

if __name__ == "__main__":
    main()
