import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def evaluate_to_dir(model, dataset, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    for videos, labels in dataset:
        probs = model.predict(videos, verbose=0)
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(probs, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()

    rows, f1s, precs, recs = [], [], [], []
    for i, cls in enumerate(class_names):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
        rows.append({'class_name': cls, 'precision': p, 'recall': r, 'f1': f1, 'support': support})

    metrics = {
        'accuracy': float(np.mean(y_true == y_pred)),
        'macro_precision': float(np.mean(precs)),
        'macro_recall': float(np.mean(recs)),
        'macro_f1': float(np.mean(f1s)),
        'support': float(len(y_true)),
    }

    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)

    with (output_dir / 'classification_report.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['class_name', 'precision', 'recall', 'f1', 'support'])
        writer.writeheader()
        writer.writerows(rows)

    np.save(output_dir / 'confusion_matrix.npy', matrix)

    # Plot normalised confusion matrix
    vals = matrix.astype(np.float32)
    row_sums = vals.sum(axis=1, keepdims=True)
    vals = np.divide(vals, row_sums, out=np.zeros_like(vals), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(vals, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('HMDB51 confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=180)
    plt.close(fig)

    return metrics


def plot_training_history(csv_path, output_path=None):
    csv_path = Path(csv_path)
    epochs, losses, val_losses, accs, val_accs = [], [], [], [], []
    with csv_path.open('r', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row['epoch']))
            losses.append(float(row['loss']))
            val_losses.append(float(row.get('val_loss', row['loss'])))
            accs.append(float(row.get('accuracy', 0)))
            val_accs.append(float(row.get('val_accuracy', 0)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, losses, label='loss')
    axes[0].plot(epochs, val_losses, label='val_loss')
    axes[0].set_title('Loss')
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, accs, label='accuracy')
    axes[1].plot(epochs, val_accs, label='val_accuracy')
    axes[1].set_title('Accuracy')
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    plt.show()


def plot_confusion_matrix(cm_path, class_names, output_path=None, sort_by_recall=True):
    """Matrice di confusione normalizzata per riga, leggibile anche con 51 classi.

    Con sort_by_recall ordina le classi da peggiore a migliore (recall crescente):
    cosi' la diagonale resta visibile e i blocchi di classi confuse emergono.
    """
    cm = np.load(cm_path).astype(np.float64)
    n = len(class_names)
    recall = np.divide(np.diag(cm), cm.sum(axis=1),
                       out=np.zeros(n), where=cm.sum(axis=1) != 0)
    order = list(np.argsort(recall)) if sort_by_recall else list(range(n))
    cm = cm[np.ix_(order, order)]
    names = [class_names[i] for i in order]

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    accuracy = np.trace(np.load(cm_path)) / max(1, np.load(cm_path).sum())

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(norm, interpolation='nearest', cmap='viridis', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='frazione predetta')
    ax.set_title(f'Matrice di confusione normalizzata — accuracy {accuracy:.3f}\n'
                 f'(classi ordinate da peggiore a migliore)')
    ax.set_xlabel('Predetto')
    ax.set_ylabel('Vero')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    plt.show()


def plot_per_class_accuracy(report_csv, output_path=None):
    """Bar chart orizzontale della recall (accuracy) per classe, ordinata.
    Colore rosso->verde: si vede subito quali azioni il modello riconosce o no."""
    with Path(report_csv).open('r', encoding='utf-8', newline='') as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r['recall']))
    names = [r['class_name'] for r in rows]
    recs = np.array([float(r['recall']) for r in rows])

    fig, ax = plt.subplots(figsize=(8, max(6, len(names) * 0.22)))
    ax.barh(np.arange(len(names)), recs, color=plt.cm.RdYlGn(recs))
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Recall (accuracy per classe)')
    ax.axvline(recs.mean(), color='black', linestyle='--', linewidth=1,
               label=f'media {recs.mean():.2f}')
    ax.set_title('Accuracy per classe (ordinata)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    plt.show()


def plot_top_confusions(cm_path, class_names, output_path=None, top_n=15):
    """Le coppie (classe vera -> classe predetta) piu' confuse: spiega DOVE sbaglia."""
    cm = np.load(cm_path).astype(np.float64)
    n = len(class_names)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    pairs = pairs[:top_n]
    labels = [f'{true} → {pred}' for _, true, pred in pairs]
    counts = [c for c, _, _ in pairs]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    ax.barh(np.arange(len(labels)), counts, color='tab:red', alpha=0.75)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Numero di video confusi')
    ax.set_title(f'Top {top_n} confusioni (classe vera → predetta)')
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=160)
    plt.show()
