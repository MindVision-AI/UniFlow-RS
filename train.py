"""Train UniFlow-RS; see uniflow_rs.training for the implementation."""


def main() -> None:
    from uniflow_rs.training.engine import main as train

    train()


if __name__ == "__main__":
    main()
