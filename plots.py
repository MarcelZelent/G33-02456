import matplotlib.pyplot as plt
import numpy as np

# Data for CPU and GPU inference times
categories = ["Baseline", "model\n5 layers", "model\n4 layers", "Student\nmodel"]
cpu_times = [16.37, 54.33, 7.21, 6.66]
cpu_quantized_times = [9.79, 10.45, 8.21, 11.88]
gpu_times = [0.87, 1.56, 1.01, 0.93]
gpu_tensorrt_times = [0.35, 0.48, 0.43, 0.32]
rpi_times = [347.6, 141.81, 101.38, 63.57]
rpi_quantized_times = [87.43, 56.67, 52.81, 38.76]

# Set global font size
plt.rcParams.update({'font.size': 14})

# Create bar positions
bar_width = 0.2
indices = np.arange(len(categories))

# CPU
plt.figure(figsize=(6, 3))
plt.bar(indices - bar_width/2, cpu_times, width=bar_width, label="CPU - Original", alpha=0.7)
plt.bar(indices + bar_width/2, cpu_quantized_times, width=bar_width, label="CPU - Quantized", alpha=0.7)
# Add labels and title
# plt.xlabel("Model Versions", fontsize=16)
plt.ylabel("Inference Time (ms)", fontsize=16)
plt.xticks(indices, categories, fontsize=14, rotation=0)  # Rotate x-axis labels
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("cpu_inference_time.png", transparent=True)

# GPU
plt.figure(figsize=(6, 3))
plt.bar(indices - bar_width/2, gpu_times, width=bar_width, label="GPU - Original", alpha=0.7)
plt.bar(indices + bar_width/2, gpu_tensorrt_times, width=bar_width, label="GPU - TensorRT", alpha=0.7)
# Add labels and title
# plt.xlabel("Model Versions", fontsize=16)
plt.ylabel("Inference Time (ms)", fontsize=16)
plt.xticks(indices, categories, fontsize=14, rotation=0)  # Rotate x-axis labels
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("gpu_inference_time.png", transparent=True)

# RPi
plt.figure(figsize=(6, 3))
plt.bar(indices - bar_width/2, rpi_times, width=bar_width, label="RPi - Original", alpha=0.7)
plt.bar(indices + bar_width/2, rpi_quantized_times, width=bar_width, label="RPi - Quantized", alpha=0.7)
# Add labels and title
# plt.xlabel("Model Versions", fontsize=16)
plt.ylabel("Inference Time (ms)", fontsize=16)
plt.xticks(indices, categories, fontsize=14, rotation=0)  # Rotate x-axis labels
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("rpi_inference_time.png", transparent=True)
