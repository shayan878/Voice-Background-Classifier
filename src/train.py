import torch
import numpy as np

def train_model(model, train_dl, valid_dl, criterion, optimizer, num_epochs=4):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        n_correct, n_total = 0, 0
        
        for inputs, targets in train_dl:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            predictions = outputs.round()
            n_correct += torch.sum(predictions == targets).item()
            n_total += targets.size(0)
            
        train_losses.append(np.mean(batch_losses))
        train_accuracies.append(n_correct / n_total)
        
        # Validation
        model.eval()
        val_batch_losses = []
        n_val_correct, n_val_total = 0, 0
        
        with torch.no_grad():
            for inputs, targets in valid_dl:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_batch_losses.append(loss.item())
                predictions = outputs.round()
                n_val_correct += torch.sum(predictions == targets).item()
                n_val_total += targets.size(0)
                
        valid_losses.append(np.mean(val_batch_losses))
        valid_accuracies.append(n_val_correct / n_val_total)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Acc: {train_accuracies[-1]:.4f} Loss: {train_losses[-1]:.4f} | "
              f"Valid Acc: {valid_accuracies[-1]:.4f} Loss: {valid_losses[-1]:.4f}")

    return train_losses, valid_losses, train_accuracies, valid_accuracies
