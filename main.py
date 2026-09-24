import torch
import torch.nn as nn
from src.data_loader import get_dataloaders
from src.model import VoiceClassifier
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # 1. Configuration
    DATA_PATH = "C:/Users/SHAYAN/Desktop/extra/voices/en"
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    EPOCHS = 4

    # 2. Data Preparation
    print("Preparing data and extracting features. This may take a moment...")
    train_dl, valid_dl, test_dl = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE)

    # 3. Model Initialization
    model = VoiceClassifier()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training
    print("\nStarting Training...")
    train_model(model, train_dl, valid_dl, criterion, optimizer, num_epochs=EPOCHS)

    # 5. Evaluation
    print("\nStarting Evaluation...")
    evaluate_model(model, test_dl, criterion)

if __name__ == "__main__":
    main()
