import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model(model, test_dl, criterion):
    model.eval()
    test_loss = []
    n_correct, n_total = 0, 0
    all_targets, all_predictions = [], []
    
    with torch.no_grad():
        for inputs, targets in test_dl:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
            
            predictions = outputs.round()
            n_correct += torch.sum(predictions == targets).item()
            n_total += targets.size(0)
            
            all_targets.extend(targets.numpy().flatten())
            all_predictions.extend(predictions.numpy().flatten())
            
    acc = n_correct / n_total
    avg_loss = np.mean(test_loss)
    
    print(f"Test Accuracy: {acc:.4f}, Test Loss: {avg_loss:.4f}")
    
    cm = confusion_matrix(all_targets, all_predictions)
    plot_confusion_matrix(cm, classes=['Voice (0)', 'Background (1)'])
