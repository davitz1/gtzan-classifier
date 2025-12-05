import torch
import numpy as np
from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).parent.parent))

from models.mfcc_cnn import MFCC_CNN
from data_processing.loader import load_and_split_data
from utils.helper_functions import (
    train, evaluate, plot_curves, get_predictions,
    plot_confusion_matrix, plot_error_analysis, 
    plot_per_class_accuracy, print_classification_report,
    plot_top_confused_pairs
)

def main():
    #reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #configs
    bs = 32
    epochs = 30
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(__file__).parent.parent.parent / "outputs" / "mfcc_cnn"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # load data
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "mfcc_data.npz"

    print("="*60)
    print("LOADING DATA")
    print("="*60)

    train_loader, val_loader, test_loader, GENRE_CLASSES = load_and_split_data(
        data_path, 
        batch_size=bs, 
        test_size=0.4,
        val_split=0.5,
        random_state=SEED,
        device=device
    )

    model = MFCC_CNN(num_classes=10, input_shape=(1, 130, 13)).to(device)
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, epochs=epochs, device=device, lr=learning_rate
    )

    # Plot curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, 
                save_path=plots_dir / "training_curves.png")

    # eval on test set 
    print("EVALUATING ON TEST SET")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    #detailed analysis
    y_true, y_pred = get_predictions(model, test_loader, device)

    #plots and reports
    cm = plot_confusion_matrix(y_true, y_pred, GENRE_CLASSES, 
                              save_path=plots_dir / "confusion_matrix.png")

    plot_per_class_accuracy(cm, GENRE_CLASSES, 
                           save_path=plots_dir / "per_class_accuracy.png")

    plot_error_analysis(cm, GENRE_CLASSES, 
                       save_path=plots_dir / "error_analysis.png")

    plot_top_confused_pairs(cm, GENRE_CLASSES, top_n=10,
                           save_path=plots_dir / "top_confused_pairs.png")

    print_classification_report(y_true, y_pred, GENRE_CLASSES,
                              save_path=output_dir / "classification_report.txt")

    #save model
    model_save_path = output_dir / "mfcc_cnn_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Save training history
    history_path = output_dir / "training_history.npz"
    np.savez(history_path,
             train_losses=train_losses,
             val_losses=val_losses,
             train_accs=train_accs,
             val_accs=val_accs)
    print(f"Training history saved to: {history_path}")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()