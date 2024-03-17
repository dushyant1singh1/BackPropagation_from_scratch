import numpy as np
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist


# Load Fashion MNIST dataset
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# Define class labels for Fashion MNIST dataset
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Initialize Weights & Biases
wandb.init(project='fashion-mnist-visualization11')

# Plot and log images to wandb
num_samples = 10  # Number of samples to display

unique_classes = np.unique(train_labels)

fig, axes = plt.subplots(1, num_samples, figsize=(100,50))
for i in range(len(unique_classes)):
    idx = np.random.randint(0, train_images.shape[0])
    axes[i].imshow(train_images[idx], cmap='gray')
    axes[i].set_title(class_labels[i])
    axes[i].axis('off')
    #wandb.log({f"Sample Image {i + 1}": [wandb.Image(train_images[idx], caption=class_labels[train_labels[idx]])]})

# Close the plot
plt.tight_layout()
wandb.log({"Sample Images": wandb.Image(fig)})
plt.close()

# Finish the run
wandb.finish()
