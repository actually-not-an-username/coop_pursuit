"""Entry point for the RMIO implementation."""


def main() -> None:
    from rmio import RMIOTrainer, select_device

    print(f"RMIO implementation ready (device: {select_device()}).")
    print("Import RMIOTrainer and the model classes from rmio.py to train an agent.")


if __name__ == "__main__":
    main()
