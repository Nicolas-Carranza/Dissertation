"""
MNIST Digit Classifier using PyTorch

This module implements a neural network classifier for the MNIST handwritten digit dataset.
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels.

Architecture:
    - Input: 28x28 pixel grayscale images (flattened to 784 features)
    - Hidden layer: 128 neurons with ReLU activation
    - Output: 10 classes (digits 0-9)

Author: Your Name
Date: October 2025
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


# =============================================================================
# MODEL DEFINITION
# =============================================================================

class MNISTClassifier(nn.Module):
    """
    A simple feedforward neural network for classifying MNIST digits.
    
    The network uses a fully connected architecture with one hidden layer.
    Input images are flattened from 28x28 to 784-dimensional vectors.
    
    Architecture:
        Input layer: 784 neurons (28x28 flattened)
        Hidden layer: 128 neurons with ReLU activation
        Output layer: 10 neurons (one per digit class)
    """
    
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        """
        Initialize the MNIST classifier.
        
        Args:
            input_size (int): Number of input features (default: 784 for 28x28 images)
            hidden_size (int): Number of neurons in hidden layer (default: 128)
            num_classes (int): Number of output classes (default: 10 for digits 0-9)
        """
        super(MNISTClassifier, self).__init__()
        
        # Flatten layer: converts 2D image (28x28) to 1D vector (784)
        self.flatten = nn.Flatten()
        
        # Sequential network with linear layers and ReLU activation
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input to hidden layer
            nn.ReLU(),                            # Non-linear activation
            nn.Linear(hidden_size, num_classes)   # Hidden to output layer
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)
        
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, 10)
        """
        x = self.flatten(x)      # Flatten: (batch, 1, 28, 28) -> (batch, 784)
        logits = self.network(x)  # Pass through network: (batch, 784) -> (batch, 10)
        return logits


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    """
    Train the model on the training dataset.
    
    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): DataLoader for training data
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimization algorithm (e.g., Adam, SGD)
        device (str): Device to train on ('cuda' or 'cpu')
        num_epochs (int): Number of training epochs
    
    Returns:
        list: Training losses for each epoch
    """
    model.train()  # Set model to training mode
    epoch_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate through batches
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Move data to device (GPU/CPU)
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Batch [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Complete - '
              f'Avg Loss: {epoch_loss:.4f}, '
              f'Accuracy: {epoch_acc:.2f}%')
        print('-' * 70)
    
    return epoch_losses


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (nn.Module): The trained neural network model
        test_loader (DataLoader): DataLoader for test data
        device (str): Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        tuple: (accuracy, average_loss)
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

def predict(model, image, device):
    """
    Make a prediction for a single image.
    
    Args:
        model (nn.Module): The trained model
        image (torch.Tensor): Input image tensor
        device (str): Device to run inference on
    
    Returns:
        tuple: (predicted_class, probabilities)
    """
    model.eval()
    
    with torch.no_grad():
        # Ensure image has correct shape: (1, 1, 28, 28)
        if image.dim() == 2:  # If shape is (28, 28)
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:  # If shape is (1, 28, 28)
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        # Get model output (logits)
        logits = model(image)
        
        # Convert to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probabilities, dim=1)
        
        return predicted_class.item(), probabilities.squeeze().cpu()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. SETUP AND CONFIGURATION
    # -------------------------------------------------------------------------
    
    # Set device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 70)
    
    # Hyperparameters
    BATCH_SIZE = 64          # Number of samples per batch
    LEARNING_RATE = 0.001    # Learning rate for optimizer
    NUM_EPOCHS = 5           # Number of training epochs
    NUM_WORKERS = 2          # Number of worker processes for data loading
    
    # -------------------------------------------------------------------------
    # 2. DATA LOADING AND PREPROCESSING
    # -------------------------------------------------------------------------
    
    # Define transformation: convert images to tensors
    # ToTensor() converts PIL images or numpy arrays to tensors and scales to [0, 1]
    transform = transforms.ToTensor()
    
    # Load MNIST training dataset
    print("Loading MNIST training dataset...")
    train_dataset = torchvision.datasets.MNIST(
        root='./data',           # Directory to store/load data
        train=True,              # Load training data
        download=True,           # Download if not present
        transform=transform      # Apply transformation
    )
    
    # Load MNIST test dataset
    print("Loading MNIST test dataset...")
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,             # Load test data
        download=True,
        transform=transform
    )
    
    # Create data loaders for batching and shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,            # Shuffle training data
        num_workers=NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,           # Don't shuffle test data
        num_workers=NUM_WORKERS
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 3. MODEL INITIALIZATION
    # -------------------------------------------------------------------------
    
    # Create model instance and move to device
    model = MNISTClassifier().to(device)
    
    print("Model Architecture:")
    print(model)
    print("=" * 70)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 4. TRAINING SETUP
    # -------------------------------------------------------------------------
    
    # Define loss function (Cross Entropy Loss for multi-class classification)
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer (Adam optimizer with learning rate)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training Configuration:")
    print(f"  Loss Function: Cross Entropy Loss")
    print(f"  Optimizer: Adam")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 5. TRAINING
    # -------------------------------------------------------------------------
    
    print("\nStarting training...")
    print("=" * 70)
    
    train_losses = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS
    )
    
    print("\nTraining completed!")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 6. EVALUATION
    # -------------------------------------------------------------------------
    
    print("\nEvaluating model on test set...")
    test_accuracy, test_loss = evaluate_model(model, test_loader, device)
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_accuracy:.2f}%")
    print(f"  Average Loss: {test_loss:.4f}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 7. SAVE MODEL
    # -------------------------------------------------------------------------
    
    # Save the trained model
    model_path = 'mnist_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_accuracy': test_accuracy,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 8. INFERENCE EXAMPLE
    # -------------------------------------------------------------------------
    
    print("\nTesting inference on a sample image...")
    
    # Get a sample image from test set
    sample_image, sample_label = test_dataset[0]
    
    # Make prediction
    predicted_class, probabilities = predict(model, sample_image, device)
    
    print(f"True Label: {sample_label}")
    print(f"Predicted Label: {predicted_class}")
    print(f"\nClass Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  Digit {i}: {prob:.4f}")
    print("=" * 70)
    

