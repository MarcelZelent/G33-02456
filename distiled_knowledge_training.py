from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import wandb

# Import your custom modules
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from loss import mse_loss
from datasets import SpectrogramDataset
from models import StudentModel, TeacherModel, weights_init_uniform_rule_stud, weights_init_uniform_rule

from torchsummary import summary

# Constants
GROUP_NUMBER = 33
LEARNING_RATE = 10e-4
EPOCHS = 2000
BATCH_SIZE = 10
NUM_WORKERS = 16  # Adjust based on your system
alpha = 0.7

# Device configuration
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = torch.device(DEVICE)

print("Device is: ", DEVICE)

# Paths
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"## Distiled Knowledge
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

def training_distiled_knowledge():

    # Prepare dataset names and directories
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    # Define transforms
    TRAIN_TRANSFORM = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])
    TEST_TRANSFORM = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "test"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])

    # Create datasets
    train_dataset = SpectrogramDataset(
        data_dir=data_dir / "train",
        stmf_data_path=DATA_ROOT / STMF_FILENAME,
        transform=TRAIN_TRANSFORM
    )
    test_dataset = SpectrogramDataset(
        data_dir=data_dir / "test",
        stmf_data_path=DATA_ROOT / STMF_FILENAME,
        transform=TEST_TRANSFORM
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize teacher model and load pre-trained weights
    teacher_model = TeacherModel().to(DEVICE)
    teacher_model.load_state_dict(torch.load(MODEL_DIR / 'model_SpectrVelCNNRegr_wobbly-rain-111'))             # MODEL_DIR / 'teacher_model.pth'
    teacher_model.eval()  # Set teacher model to evaluation mode

    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize student model
    student_model = StudentModel().to(DEVICE)
    student_model.apply(weights_init_uniform_rule_stud)  # Initialize weights

    # Define Distillation Loss Function
    class DistillationLossRegression(nn.Module):
        def __init__(self, alpha=0.5):
            super(DistillationLossRegression, self).__init__()
            self.alpha = alpha
            self.mse_loss = nn.MSELoss()

        def forward(self, student_preds, teacher_preds, true_targets):
            # MSE loss for hard labels
            hard_loss = self.mse_loss(student_preds, true_targets)

            # MSE loss for soft labels
            soft_loss = self.mse_loss(student_preds, teacher_preds)

            # Combined Loss
            combined_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            return combined_loss, hard_loss.item(), soft_loss.item()

    # Optimizer for Student Model
    optimizer = torch.optim.Adam(student_model.parameters(), lr=LEARNING_RATE)

    # Loss function
    distillation_loss_fn = DistillationLossRegression(alpha=alpha)
    student_loss_fn = nn.MSELoss()

    def train_one_epoch(student_model, teacher_model, data_loader, optimizer, device, distillation_loss_fn):
        student_model.train()
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0

        for data in data_loader:
            spectrogram, target = data["spectrogram"].to(device), data["target"].to(device)

            optimizer.zero_grad()

            # Forward pass
            student_outputs = student_model(spectrogram)
            with torch.no_grad():
                teacher_outputs = teacher_model(spectrogram)

            # Compute combined loss and individual losses
            loss, hard_loss, soft_loss = distillation_loss_fn(student_outputs.squeeze(), teacher_outputs.squeeze(), target)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_hard_loss += hard_loss
            total_soft_loss += soft_loss

        avg_loss = total_loss / len(data_loader)
        avg_hard_loss = total_hard_loss / len(data_loader)
        avg_soft_loss = total_soft_loss / len(data_loader)
        return avg_loss, avg_hard_loss, avg_soft_loss

    def validate(student_model, data_loader, device, loss_fn):
        student_model.eval()
        total_loss = 0

        with torch.no_grad():
            for data in data_loader:
                spectrogram, target = data["spectrogram"].to(device), data["target"].to(device)
                outputs = student_model(spectrogram)
                loss = loss_fn(outputs.squeeze(), target)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        return avg_loss

    wandb.init(
        project=f"02456_group_{GROUP_NUMBER}",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": student_model.__class__.__name__,
            "dataset": "SpectrogramDataset",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "loss_fn": "MSE",
            "alpha": alpha,
            "nfft": NFFT
        }
    )

    # Define model path for saving
    MODEL_DIR.mkdir(exist_ok=True)
    model_name = f"student_model_{wandb.run.name}.pth"
    model_path = MODEL_DIR / model_name

    best_vloss = float('inf')

    model_sum = StudentModel().to(DEVICE)  # or YourModel()
    input_size = (6, 74, 918)  # Replace H and W with your input dimensions
    summary(model_sum, input_size=input_size)

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')

        # Train the student model for one epoch
        avg_train_loss, avg_hard_loss, avg_soft_loss = train_one_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            distillation_loss_fn=distillation_loss_fn
        )

        # Validate the student model
        avg_val_loss = validate(
            student_model=student_model,
            data_loader=test_loader,
            device=DEVICE,
            loss_fn=student_loss_fn
        )

        # Calculate RMSE
        train_rmse = avg_train_loss ** 0.5
        val_rmse = avg_val_loss ** 0.5

        print(f'EPOCH {epoch + 1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Log metrics to WandB
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_rmse': train_rmse,
            'test_loss': avg_val_loss,
            'test_rmse': val_rmse,
            'train_hard_loss': avg_hard_loss,
            'train_soft_loss': avg_soft_loss
        })

        # Save the best model
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            torch.save(student_model.state_dict(), model_path)

    wandb.finish()


if __name__ == "__main__":
    training_distiled_knowledge()
