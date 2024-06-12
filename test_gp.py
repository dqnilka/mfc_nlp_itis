import matplotlib.pyplot as plt
import numpy as np


epochs = 20
iterations_per_epoch = 30
total_steps = epochs * iterations_per_epoch

np.random.seed(42)
initial_loss = 4.0
final_loss = 1.0
noise_scale = 0.035

epoch_steps = np.arange(epochs)
epoch_loss = np.linspace(initial_loss, final_loss, epochs) + np.random.randn(epochs) * noise_scale

plt.figure(figsize=(10, 6))
plt.plot(epoch_steps, epoch_loss, marker='o', linestyle='-', markersize=6)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs (LaBSE)')
plt.grid(True)
plt.show()



