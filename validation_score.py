from pathlib import Path
import torch
import time
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from models import SpectrVelCNNRegr
from data_management import make_dataset_name
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from torch.quantization import quantize_fx

# CONSTANTS TO MODIFY AS YOU WISH
# MODEL = SpectrVelCNNRegr
DEVICE = (
    "cpu"
    # "cuda"
    # if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    # else "cpu"
)

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
# LOAD_ORIGINAL_MODEL_FNAME = "model_YourModel_glowing-paper-164.pth"  # Original model file
# LOAD_QUANTIZED_MODEL_FNAME = LOAD_ORIGINAL_MODEL_FNAME + "_quantized"  # Quantized model file

def validate_quant(model_class=SpectrVelCNNRegr, weights_path_original="baseline.pth", weights_path_quantized="baseline.pth_quantized"):

    def evaluate_model(model, data_loader):
        """Evaluate the model on the validation dataset and return metrics."""
        running_val_loss = torch.tensor(0.0, device=DEVICE)

        with torch.no_grad():
            for vdata in data_loader:
                spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                val_outputs = model(spectrogram)
                val_loss = MODEL.loss_fn(val_outputs.squeeze(), target)
                running_val_loss += val_loss

        avg_val_loss = running_val_loss / len(data_loader)
        val_rmse = avg_val_loss**(1/2)
        log_val_rmse = torch.log10(val_rmse)
        return avg_val_loss.item(), val_rmse.item(), log_val_rmse.item()

    def measure_inference_time(model, data_loader, num_batches=80):
        """
        Measure the average inference time per batch for a given model and data loader.
        Args:
            model: The model to evaluate.
            data_loader: DataLoader providing input data.
            num_batches: Number of batches to use for timing.
        Returns:
            Average inference time per batch in milliseconds.
        """
        model.eval()  # Ensure the model is in evaluation mode
        total_time = 0.0
        num_measured_batches = 0

        with torch.no_grad():  # Disable gradient computation
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break  # Limit to num_batches for timing

                spectrogram = batch["spectrogram"].to("cpu")  # Ensure CPU for quantized model
                start_time = time.time()
                _ = model(spectrogram)  # Perform forward pass
                end_time = time.time()

                total_time += (end_time - start_time)
                num_measured_batches += 1

        avg_time_per_batch = (total_time / num_measured_batches) * 1000  # Convert to milliseconds
        return avg_time_per_batch


    MODEL = model_class
    LOAD_ORIGINAL_MODEL_FNAME = weights_path_original
    LOAD_QUANTIZED_MODEL_FNAME = weights_path_quantized

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
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)

    # Load and evaluate the original model
    original_model = MODEL().to(DEVICE)
    original_model.load_state_dict(torch.load(MODEL_DIR / LOAD_ORIGINAL_MODEL_FNAME, map_location=DEVICE))
    original_model.eval()


    # Load and evaluate the quantized model
    torch.backends.quantized.engine = "fbgemm" # Use "qnnpack" for ARM
    quantized_model = MODEL().to("cpu")  # Quantized models typically run on CPU
    quantized_model.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig("fbgemm")} # "qnnpack" for ARM
    quantized_model = quantize_fx.prepare_fx(quantized_model, qconfig_dict, example_inputs=next(iter(validation_data_loader))["spectrogram"])
    quantized_model = quantize_fx.convert_fx(quantized_model)
    quantized_model.load_state_dict(torch.load(MODEL_DIR / LOAD_QUANTIZED_MODEL_FNAME, map_location="cpu"))

    def count_quantized_params(model):
        total_params = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight()
                total_params += weight.numel()
                # print(f"{name} weight: {weight.numel()} parameters")
            if hasattr(module, 'bias') and module.bias() is not None:
                bias = module.bias()
                total_params += bias.numel()
                # print(f"{name} bias: {bias.numel()} parameters")
        return total_params

    quantized_params = count_quantized_params(quantized_model)

    print("Evaluating Original Model...")
    original_loss, original_rmse, original_log_rmse = evaluate_model(original_model, validation_data_loader)
    print("Measuring inference time for the original model...")
    original_time = measure_inference_time(original_model, validation_data_loader)
    print(f"Original Model - Average Inference Time per Sample: {original_time:.2f} ms")
    print(f"Original Model - Validation Loss: {original_loss}")
    print(f"Original Model - Validation RMSE: {original_rmse}")
    print(f"Original Model - Log10 Validation RMSE: {original_log_rmse}")
    print(f"Original Model - Number of parameters: {sum(p.numel() for p in original_model.parameters())}")
    print(f"Original Model - Size of model: {Path(MODEL_DIR / LOAD_ORIGINAL_MODEL_FNAME).stat().st_size / 1e6:.2f} MB")

    print("Evaluating Quantized Model...")
    quantized_loss, quantized_rmse, quantized_log_rmse = evaluate_model(quantized_model, validation_data_loader)
    print("Measuring inference time for the quantized model...")
    quantized_time = measure_inference_time(quantized_model, validation_data_loader)
    print(f"Quantized Model - Average Inference Time per Sample: {quantized_time:.2f} ms, speedup: {original_time / quantized_time:.2f}")
    print(f"Quantized Model - Validation Loss: {quantized_loss}")
    print(f"Quantized Model - Validation RMSE: {quantized_rmse}")
    print(f"Quantized Model - Log10 Validation RMSE: {quantized_log_rmse}")
    print(f"Quantized Model - Number of parameters: {quantized_params}, difference: {sum(p.numel() for p in original_model.parameters()) - quantized_params}")
    print(f"Quantized Model - Size of model: {Path(MODEL_DIR / LOAD_QUANTIZED_MODEL_FNAME).stat().st_size / 1e6:.2f} MB")
    print(f"Quantized Model - Original Model Size Ratio: {Path(MODEL_DIR / LOAD_QUANTIZED_MODEL_FNAME).stat().st_size / Path(MODEL_DIR / LOAD_ORIGINAL_MODEL_FNAME).stat().st_size:.2f}")


if __name__ == "__main__":
    validate_quant()
