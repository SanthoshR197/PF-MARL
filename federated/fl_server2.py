import flwr as fl

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    print("Starting FL Server 2 (DP PF-MARL) on Port 8083...")
    fl.server.start_server(
        server_address="0.0.0.0:8083",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
