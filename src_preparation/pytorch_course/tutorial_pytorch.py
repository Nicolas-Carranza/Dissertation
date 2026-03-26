import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from pathlib import Path


def first_part():
    '''
    This function is part of the PyTorch tutorial.
    
    It demonstrates tensor creation, properties, operations, and the bridge with NumPy.
        1. Tensor Creation
        2. Tensor Properties
        3. Tensor Operations
        4. Bridge with NumPy
    '''
    print("\n" + "=" * 70)
    print("PART 1: Tensors and NumPy bridge")
    print("=" * 70)

    # Create a tensor from data
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print(f"\n[1] Tensor from Python list:\n{x_data}\n")

    # Create a tensor from a NumPy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print(f"[2] Tensor from NumPy array:\n{x_np}\n")

    # Create a tensor from another tensor
    x_ones = torch.ones_like(x_data)  # retains the properties of x_data
    print(f"[3] Ones tensor (same shape as x_data):\n{x_ones}\n")

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
    print(f"[4] Random tensor (float):\n{x_rand}\n")

    # Create random, ones, and zeros tensors
    shape = (2,3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"[5] Random tensor:\n{rand_tensor}\n")
    print(f"[6] Ones tensor:\n{ones_tensor}\n")
    print(f"[7] Zeros tensor:\n{zeros_tensor}")

    # Get tensor properties
    tensor = torch.rand(3,4)

    print("\n[8] Tensor properties")
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")

    # We can move our tensor to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to(torch.accelerator.current_accelerator())
        print(f"Moved tensor to accelerator: {tensor.device}")

    # Tensor operations
    # Standard numpy-like indexing and slicing
    tensor = torch.ones(4,4)
    print("\n[9] Indexing and slicing")
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:,0]}")
    print(f"Last column: {tensor[:,-1]}")
    tensor[:,1] = 0
    print(f"Modified tensor:\n{tensor}")

    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(f"Concatenated tensor:\n{t1}")

    # Arithmetic operations
    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    y1 = tensor @ tensor.T
    y2 = tensor.matmul(tensor.T)
    y3 = torch.rand_like(tensor)
    torch.matmul(tensor, tensor.T, out=y3)
    print(f"\n[10] Matrix multiplication result:\n{y1}\n")

    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = tensor * tensor
    z2 = tensor.mul(tensor)
    z3 = torch.rand_like(tensor)

    torch.mul(tensor, tensor, out=z3)
    print(f"[11] Element-wise multiplication result:\n{z1}\n")

    agg = tensor.sum()
    agg_item = agg.item()
    print(f"[12] Sum converted to Python scalar: {agg_item} ({type(agg_item).__name__})")

    print(f"\nTensor before in-place add:\n{tensor}\n")
    tensor.add_(5)
    print(f"Tensor after add_(5):\n{tensor}")
    
    # Bridge with NumPy
    t = torch.ones(5)
    print("\n[13] PyTorch <-> NumPy memory sharing")
    print(f"t: {t}")
    n = t.numpy()
    print(f"n: {n}")

    t.add_(1)
    print(f"t after add_: {t}")
    print(f"n after t is modified: {n}")

    n = np.ones(5)
    t = torch.from_numpy(n)

    np.add(n, 1, out=n)
    print(f"n after np.add: {n}")
    print(f"t after n is modified: {t}")


def _show_or_save_figure(figure, output_name):
    """Show plot on desktop sessions; save to disk on headless sessions."""
    if os.environ.get("DISPLAY"):
        plt.tight_layout()
        plt.show()
        return None

    output_path = Path(__file__).resolve().parent / output_name
    plt.tight_layout()
    figure.savefig(output_path, dpi=150)
    print(f"No DISPLAY detected. Saved figure to: {output_path}")
    return output_path

def second_part():
    '''
    This function is part of the PyTorch tutorial.
    
    
    '''
    print("\n" + "=" * 70)
    print("PART 2: FashionMNIST sample visualization")
    print("=" * 70)

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    labels_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    
    figure = plt.figure(figsize=(8,8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_index = torch.randint(len(training_data), size=(1,)).item()
        
        img, label = training_data[sample_index]
        
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    print(f"Loaded training samples: {len(training_data)}")
    print(f"Loaded test samples: {len(test_data)}")
    _show_or_save_figure(figure, "fashion_mnist_samples.png")

if __name__ == "__main__":
    
    # Prompt the user to select which part to run
    part = input("Enter '1' for tensor basics or '2' for FashionMNIST visualization: ").strip()
    
    if part == '1':
        first_part()
    elif part == '2':
        second_part()
    else:
        print("Invalid selection. Please run again and choose '1' or '2'.")
    