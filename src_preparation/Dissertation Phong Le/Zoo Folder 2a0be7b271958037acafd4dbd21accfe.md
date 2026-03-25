# Zoo Folder

Reconstruction game

| Run | Mode | Architecture | LR | MaxLen | Vocab | Epochs | **Final Acc** | Msg Len | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | GS | S:256/R:512 GRU | 0.01 | 4 | 100 | 50 | **3%** | 2.09 | Too high LR |
| **2** | GS | Mismatch | 0.01 | 4 | 100 | - | **CRASH** | - | Data error |
| **3** | GS | S:256/R:512 GRU | 0.001 | 6 | 100 | 200 | **14%** | 5.39 | Better |
| **4** | **RF** | S:256/R:512 GRU | 0.001 | 6 | 100 | 200 | **17%** ⭐ | 7.0 | Best |
| **5** | GS | S:128/R:128 GRU | 0.0005 | 6 | 50 | 200 | **5%** | 2.98 | Too small |
| **6** | GS | S:256/R:512 **LSTM** | 0.001 | 7 | 20 | 200 | **7%** | 4.24 | Slower |

# EGG Basic Games: Complete System Specification

## Overview

The EGG basic games module implements two fundamental signaling games for studying emergent communication: reconstruction and discrimination. The system consists of neural agents (Sender and Receiver) that learn to communicate through differentiable training methods.

## System Architecture

### Core Components

**Data Processing Layer** (data_readers.py)

- Two PyTorch Dataset classes handle input data transformation:
    - `AttValRecoDataset`: Processes reconstruction game data. Reads space-delimited attribute-value vectors from text files. Converts integer vectors to one-hot encoding. Returns tuples containing sender_input (one-hot tensor) and labels (integer tensor). Both fields contain identical information in different formats.
    - `AttValDiscriDataset`: Processes discrimination game data. Reads period-delimited fields where each field (except last) represents an attribute-value vector. Final field indicates target vector index (zero-based). Returns tuples containing sender_input (target as one-hot), labels (target index), and receiver_input (matrix of all candidates as one-hot).

**Agent Architectures** (`architectures.py`)

- Three neural network modules define agent core functionality:
    - `Sender`: Universal sender architecture for both games. Single linear layer maps input features to hidden representation. Hidden layer initialises message-generating RNN in wrapper. Applies linear transformation: input → hidden state.
    - `RecoReceiver`: Receiver for reconstruction game. Single linear layer transforms RNN output to feature space. Maps hidden representation to n_features dimensionality. Output interpreted as reconstructed attribute-value vector.
    - `DiscriReceiver`: Receiver for discrimination game. Linear layer embeds candidate vectors to hidden dimensionality. Applies tanh nonlinearity to embedded candidates. Computes dot products between RNN output and each candidate. Returns unnormalised probability distribution over candidate positions.

**Training Pipeline** (`play.py`)

- Main script orchestrating game execution and training.

## Game Specifications

Reconstruction Game:

- **Objective**:
    - Sender communicates complete attribute-value vector. Receiver reconstructs all attributes from message alone.
- **Data Format**:
    - Text file with one vector per line. Space-delimited integer values. Example: `0 1 2` for three attributes.
- **Loss Function**:
    - Cross-entropy computed independently per attribute. Averaged across all attributes. Formula: mean cross-entropy over n_attributes predictions versus ground truth.
- **Accuracy Metric**:
    - Binary per-sample metric. Success requires all attributes predicted correctly. Partial reconstruction counts as failure.
- **Input/Output Flow**:
    - Sender receives one-hot encoded vector of size (n_attributes × n_values). Sender produces variable-length message. Receiver processes message only (no additional input). Receiver outputs n_attributes predictions, each over n_values possibilities.

Discrimination Game

- **Objective**:
    - Sender identifies target among distractors. Receiver determines target position from message and candidates.
- **Data Format**:
    - Text file with period-delimited fields. Multiple candidate vectors separated by periods. Final field contains target index. Example: `1 2.0 1.0 3.0` where 0 indicates first vector is target.
- **Loss Function**:
    - Cross-entropy between predicted position distribution and true target index. Single classification task per sample.
- **Accuracy Metric**:
    - Binary per-sample metric. Success requires correct identification of target position. Computed as argmax(receiver_output) == target_index.
- **Input/Output Flow**:
    - Sender receives target vector as one-hot encoding. Sender produces variable-length message. Receiver processes message plus matrix of all candidates. Receiver outputs probability distribution over candidate positions.

## Training Methods

Gumbel-Softmax Mode (--mode gs)

- **Communication Type**:
    - Continuous relaxation of discrete symbols. Differentiable throughout computation graph.
- **Sender Wrapper**:
    - `RnnSenderGS`. Produces soft probability distributions over vocabulary. Uses temperature parameter for sharpness control. Temperature anneals during training (decay 0.9, minimum 0.1).
- **Receiver Wrapper**:
    - `RnnReceiverGS`. Processes continuous symbol representations. Applies RNN over soft distributions.
- **Game Object**:
    - `SenderReceiverRnnGS`. Enables end-to-end backpropagation. No sampling required during training.
- **Callbacks**:
    - `TemperatureUpdater` progressively sharpens distributions.

REINFORCE Mode (--mode rf)

- **Communication Type**:
    - Discrete symbol sampling. Policy gradient optimisation.
- **Sender Wrapper**:
    - `RnnSenderReinforce`. Samples discrete symbols from learned policy. Requires entropy regularisation for exploration. Default entropy coefficient: 0.1.
- **Receiver Wrapper**:
    - `RnnReceiverDeterministic`. Processes discrete symbol sequences. Uses standard RNN processing.
- **Game Object**:
    - `SenderReceiverRnnReinforce`. Applies REINFORCE algorithm. Computes policy gradients via sampling.
- **Callbacks**:
    - None by default.

## Configuration Parameters

Game Selection

- `game_type`: reco (reconstruction) or discri (discrimination).
- Default: reco.

Data Parameters

- `train_data`: Path to training dataset file. Required.
- `validation_data`: Path to validation dataset file. Required.
- `n_attributes`: Number of object attributes. Required for reconstruction game only.
- `n_values`: Number of possible values per attribute. Required for both games.
- `batch_size`: Training batch size. Inherited from EGG core parameters.
- `validation_batch_size`: Validation batch size. Default: same as training batch size.

Training Method

- `mode`: Training algorithm (gs or rf). Default: rf.
- `temperature`: Gumbel-Softmax initial temperature. Default: 1.0. Only for gs mode.
- `sender_entropy_coeff`: Entropy regularisation coefficient. Default: 0.1. Only for rf mode.
- `n_epochs`: Number of training epochs. Inherited from EGG core.
- `lr`: Learning rate. Inherited from EGG core.

Agent Architecture

- `sender_cell`: RNN type for sender (rnn, gru, lstm). Default: rnn.
- `receiver_cell`: RNN type for receiver (rnn, gru, lstm). Default: rnn.
- `sender_hidden`: Sender hidden layer size. Default: 10.
- `receiver_hidden`: Receiver hidden layer size. Default: 10.
- `sender_embedding`: Sender symbol embedding dimension. Default: 10.
- `receiver_embedding`: Receiver symbol embedding dimension. Default: 10.
- `vocab_size`: Vocabulary size for messages. Inherited from EGG core.
- `max_len`: Maximum message length. Inherited from EGG core.

Output Control

- `print_validation_events`: Print detailed validation results after training.
- Default: False.

## Training Execution Flow

**Initialisation Phase**:

- Parse command line arguments and inherit EGG core parameters.
- Set validation batch size to training batch size if not specified.
- Select game-specific components based on game_type parameter.

**Data Loading Phase**:

- Instantiate appropriate Dataset class for chosen game.
- Create PyTorch DataLoader for training data with shuffling enabled.
- Create PyTorch DataLoader for validation data without shuffling.
- Extract feature dimensionality from processed data.

**Architecture Construction Phase**:

- Instantiate core agent modules (Sender and game-specific Receiver).
- Wrap core modules in training-mode-specific wrappers (GS or REINFORCE).
- Combine wrapped agents and loss function into Game object.
- Initialise optimizer with game parameters.
- Configure callbacks (TemperatureUpdater for GS, ConsoleLogger always).

**Training Phase**:

- Trainer object manages training loop over specified epochs.
- Each epoch processes all training batches with gradient updates.
- Validation performed after each epoch without gradient computation.
- ConsoleLogger outputs loss and accuracy in JSON format.
- Optional PrintValidationEvents callback shows detailed validation results.

## Data Flow During Training

**Forward Pass**:

- Batch loaded from DataLoader containing sender_input, labels, and optionally receiver_input.
- Sender processes input vector through core module producing hidden state.
- Sender wrapper uses hidden state to initialise RNN generating message sequence.
- Message passed to receiver wrapper which processes it through RNN producing hidden state.
- Receiver core module combines RNN output with game-specific input producing final output.
- Loss function computes task-specific loss and auxiliary metrics (accuracy).

**Backward Pass**:

- Loss gradients backpropagated through entire computation graph.
- In GS mode: standard backpropagation through continuous operations.
- In REINFORCE mode: policy gradients computed via sampling and reward weighting.
- Optimizer updates all trainable parameters.
- Callbacks execute post-epoch operations (temperature annealing, logging).

Key Design Principles

- **Modularity**:
    - Core agent architectures separated from training-specific wrappers. Same core modules usable with different optimisation methods.
- **Flexibility**:
    - Extensive configuration options for experimentation. Game-agnostic core framework with game-specific extensions.
- **Standard Integration**:
    - Full PyTorch compatibility using standard components. DataLoader for batching, standard optimisers, F.cross_entropy for loss.
- **Extensibility**:
    - Callback system allows custom training modifications. Easy addition of new game types following existing patterns.

## Example Usage

**Reconstruction Game with Gumbel-Softmax**:

```
python -m egg.zoo.basic_games.play --mode gs --game_type reco --train_data data/train.txt --validation_data data/val.txt --n_attributes 5 --n_values 3 --n_epochs 50 --batch_size 32 --vocab_size 100 --max_len 4 --sender_hidden 256 --receiver_hidden 512 --sender_embedding 10 --receiver_embedding 30 --temperature 1.0 --lr 0.001

```

**Discrimination Game with REINFORCE**:

```
python -m egg.zoo.basic_games.play --mode rf --game_type discri --train_data data/train.txt --validation_data data/val.txt --n_values 3 --n_epochs 100 --batch_size 64 --vocab_size 50 --max_len 5 --sender_hidden 128 --receiver_hidden 256 --sender_entropy_coeff 0.1 --lr 0.01 --print_validation_events

```