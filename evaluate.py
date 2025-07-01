import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from datasets import TreeSubsetDataset, get_transform
from models import get_model

def evaluate(model, dataloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            loss = criterion(output, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels

def per_class_accuracy(y_true, y_pred, num_classes):
    correct = Counter()
    total = Counter()
    for yt, yp in zip(y_true, y_pred):
        total[yt] += 1
        if yt == yp:
            correct[yt] += 1
    return [correct[i] / total[i] if total[i] > 0 else 0 for i in range(num_classes)]

def save_results(output_dir, name, cm, loss, acc, class_acc):
    np.save(os.path.join(output_dir, f"confusion_{name}.npy"), cm)
    return {
        f"{name}_loss": float(loss),
        f"{name}_accuracy": float(acc),
        f"{name}_class_accuracy": [float(x) for x in class_acc]
    }

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.experiment_dir, "metadata.json")) as f:
        meta = json.load(f)

    transform = get_transform(model_type=meta["model"])
    test_ds = TreeSubsetDataset(meta["dataset"], root="./data", train=False,
                                 transform=transform, class_names=meta["classes"])
    test_loader = DataLoader(test_ds, batch_size=32)

    num_classes = len(meta["classes"])

    fed_model = get_model(meta["model"], num_classes, pretrained=False)
    fed_model.load_state_dict(torch.load(os.path.join(args.experiment_dir, "fed_model.pt"), map_location=device))
    fed_model.to(device)

    fed_loss, fed_acc, fed_preds, fed_labels = evaluate(fed_model, test_loader, device)
    fed_class_acc = per_class_accuracy(fed_labels, fed_preds, num_classes)
    cm_fed = confusion_matrix(fed_labels, fed_preds, labels=range(num_classes), normalize='true')

    central_model = get_model(meta["model"], num_classes, pretrained=False)
    central_model.load_state_dict(torch.load(os.path.join(args.experiment_dir, "central_model.pt"), map_location=device))
    central_model.to(device)

    central_loss, central_acc, central_preds, central_labels = evaluate(central_model, test_loader, device)
    central_class_acc = per_class_accuracy(central_labels, central_preds, num_classes)
    cm_central = confusion_matrix(central_labels, central_preds, labels=range(num_classes), normalize='true')

    print(f"\nFederated   -> Loss: {fed_loss:.4f} | Accuracy: {fed_acc*100:.2f}%")
    print(f"Centralized -> Loss: {central_loss:.4f} | Accuracy: {central_acc*100:.2f}%\n")

    print("Per-Class Accuracy:")
    for i, cls in enumerate(meta["classes"]):
        f_acc = fed_class_acc[i] * 100
        c_acc = central_class_acc[i] * 100
        print(f"{cls:15s} | Fed: {f_acc:.2f}% | Central: {c_acc:.2f}%")

    results = {}
    results.update(save_results(args.experiment_dir, "fed", cm_fed, fed_loss, fed_acc, fed_class_acc))
    results.update(save_results(args.experiment_dir, "central", cm_central, central_loss, central_acc, central_class_acc))

    with open(os.path.join(args.experiment_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    if args.plot:
        plots_dir = os.path.join(args.experiment_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        x = np.arange(num_classes)
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, fed_class_acc, width, label='Federated')
        plt.bar(x + width/2, central_class_acc, width, label='Centralized')
        plt.xticks(x, meta["classes"], rotation=45)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Class-wise Accuracy")
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "class_accuracy.png"))
        plt.close()

        fig1, ax1 = plt.subplots(figsize=(6, 6))
        disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_fed, display_labels=meta["classes"])
        disp1.plot(ax=ax1, xticks_rotation=45)
        plt.title("Federated Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_fed.png"))
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 6))
        disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_central, display_labels=meta["classes"])
        disp2.plot(ax=ax2, xticks_rotation=45)
        plt.title("Centralized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_central.png"))
        plt.close(fig2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    main(args)
