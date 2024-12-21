import torch
import time
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from models import SpectrVelCNNRegr
from data_management import make_dataset_name
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from torch_tensorrt import compile, Input

# CONSTANTS TO MODIFY AS YOU WISH
# MODEL = SpectrVelCNNRegr
DEVICE = "cuda" # only cuda
BATCH_SIZE = 1

# Paths and dataset constants
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data")
# DATA_ROOT = Path(f"/home/filip/projects/DTU/deep_learning/deep_learning_project_4/data")
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

# File names for models
# LOAD_ORIGINAL_MODEL_FNAME = "model_SpectrVelCNNRegr_dauntless-capybara-82.pth"  # Original model file


def validate_quant_dynamic(model_class=SpectrVelCNNRegr, weights_path_original=Path("baseline.pth")):

    def evaluate_model(model, data_loader):
        """Evaluate the model on the validation dataset and return metrics."""
        model.eval()
        running_val_loss = torch.tensor(0.0, device=DEVICE)

        with torch.no_grad():
            for vdata in data_loader:
                spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                val_outputs = model(spectrogram)
                val_loss = MODEL.loss_fn(val_outputs.squeeze(), target)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / len(data_loader)
        val_rmse = avg_val_loss**(1 / 2)
        log_val_rmse = torch.log10(val_rmse)
        return avg_val_loss.item(), val_rmse.item(), log_val_rmse.item()

    def measure_inference_time(model, data_loader, num_batches=80):
        """
        Measure the average inference time per batch for a given model and data loader.
        """
        model.eval()
        total_time = 0.0
        num_measured_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break  # Limit to num_batches for timing

                spectrogram = batch["spectrogram"].to(DEVICE)
                start_time = time.time()
                _ = model(spectrogram)  # Perform forward pass
                end_time = time.time()

                total_time += (end_time - start_time)
                num_measured_batches += 1

        avg_time_per_batch = (total_time / num_measured_batches) * 1000  # Convert to milliseconds
        return avg_time_per_batch


    MODEL = model_class
    LOAD_ORIGINAL_MODEL_FNAME = weights_path_original

    print(f"Using {DEVICE} device")

    # DATA SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    VALIDATION_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "validation"),
         NormalizeSpectrogram(),
         ToTensor(),
         InterpolateSpectrogram()]
    )
    dataset_validation = MODEL.dataset(data_dir=data_dir / "validation",
                                       stmf_data_path=DATA_ROOT / STMF_FILENAME,
                                       transform=VALIDATION_TRANSFORM)

    validation_data_loader = DataLoader(dataset_validation,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)

    # Load the original model
    original_model = MODEL().to(DEVICE)
    original_model.load_state_dict(torch.load(MODEL_DIR / LOAD_ORIGINAL_MODEL_FNAME, map_location=DEVICE))
    original_model.eval()

    print("Evaluating Original Model...")
    original_loss, original_rmse, original_log_rmse = evaluate_model(original_model, validation_data_loader)
    print("Measuring inference time for the original model...")
    original_time = measure_inference_time(original_model, validation_data_loader)
    print(f"Original Model - Average Inference Time per Batch: {original_time:.2f} ms")
    print(f"Original Model - Validation Loss: {original_loss}")
    print(f"Original Model - Validation RMSE: {original_rmse}")
    print(f"Original Model - Log10 Validation RMSE: {original_log_rmse}")

    # Optimize model with TensorRT
    print("Optimizing Model with TensorRT...")
    input_spec = Input(
        dtype=torch.float32,
        shape=(BATCH_SIZE, *next(iter(validation_data_loader))["spectrogram"].shape[1:]),
        # min_shape=(1, *next(iter(validation_data_loader))["spectrogram"].shape[1:]),
        # opt_shape=(BATCH_SIZE, *next(iter(validation_data_loader))["spectrogram"].shape[1:]),
        # max_shape=(BATCH_SIZE * 2, *next(iter(validation_data_loader))["spectrogram"].shape[1:]),
    )

    trt_model = compile(
        original_model,
        inputs=[input_spec],
        enabled_precisions={torch.half, torch.float32},
    )
    print("Evaluating TensorRT Optimized Model...")
    trt_loss, trt_rmse, trt_log_rmse = evaluate_model(trt_model, validation_data_loader)
    print("Measuring inference time for the TensorRT optimized model...")
    trt_time = measure_inference_time(trt_model, validation_data_loader)
    print(f"TensorRT Model - Average Inference Time per Batch: {trt_time:.2f} ms")
    print(f"TensorRT Model - Validation Loss: {trt_loss}")
    print(f"TensorRT Model - Validation RMSE: {trt_rmse}")
    print(f"TensorRT Model - Log10 Validation RMSE: {trt_log_rmse}")
    print(f"TensorRT Speedup: {original_time / trt_time:.2f}")


if __name__ == "__main__":
    validate_quant_dynamic()
