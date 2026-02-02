from src.train import train_classifier
from src.evaluate import run_evaluation
from src.autoencoder import train_autoencoder


def main():
    print("=" * 60)
    print("STAGE 1: Training ResNet18 classifier")
    print("=" * 60)
    model, test_loader, device = train_classifier()

    print("\n" + "=" * 60)
    print("STAGE 2: Evaluating classifier")
    print("=" * 60)
    run_evaluation(model, test_loader, device)

    print("\n" + "=" * 60)
    print("STAGE 3: Training autoencoder on good parts")
    print("=" * 60)
    train_autoencoder()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
