import torch
import copy
from torch.quantization import quantize_fx
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from models import SpectrVelCNNRegr
from data_management import make_dataset_name
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram


# LOAD_MODEL_FNAME = Path("model_SpectrVelCNNRegr_wobbly-rain-111.pth")
# MODEL = SpectrVelCNNRegr

DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data")
# DATA_ROOT = Path(f"/home/filip/projects/DTU/deep_learning/deep_learning_project_4/data")
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)


def quantize_model(model_class=SpectrVelCNNRegr, weights_path=Path("baseline.pth")):
    MODEL = model_class
    LOAD_MODEL_FNAME = weights_path
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    VALIDATION_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )

    dataset_validation = MODEL.dataset(data_dir= data_dir / "train",
                        stmf_data_path = DATA_ROOT / STMF_FILENAME,
                        transform=VALIDATION_TRANSFORM)

    validation_data_loader = DataLoader(dataset_validation,
                                batch_size=500,
                                shuffle=False,
                                num_workers=4)

    # Ensure the model is loaded correctly
    if LOAD_MODEL_FNAME is not None:
        model = MODEL()
        model.load_state_dict(torch.load(MODEL_DIR / LOAD_MODEL_FNAME))
        model.eval()
    else:
        raise ValueError("Please specify LOAD_MODEL_FNAME to load a trained model for quantization.")

    # Move model to CPU (quantization is CPU-specific)
    model.to("cpu")

    # Clone the model for quantization
    m = copy.deepcopy(model)
    m.eval()

    # Define backend for quantization
    backend = "fbgemm"  # 'qnnpack' is another option, depending on your hardware
    torch.backends.quantized.engine = backend

    # Create a qconfig dictionary
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}

    # Prepare example inputs for calibration
    example_inputs = next(iter(validation_data_loader))["spectrogram"]

    # Prepare the model for quantization
    model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, example_inputs)

    # Calibrate using representative validation data
    print("Calibrating the model...")
    with torch.inference_mode():
        for i, data in tqdm(enumerate(validation_data_loader), desc="Calibrating"):
            spectrogram, target = data["spectrogram"], data["target"]
            model_prepared(spectrogram)

    # Convert the model to quantized form
    print("Converting the model to quantized form...")
    model_quantized = quantize_fx.convert_fx(model_prepared)

    # Save the quantized model
    quantized_model_name = f"{LOAD_MODEL_FNAME}_quantized"
    quantized_model_path = MODEL_DIR / quantized_model_name
    torch.save(model_quantized.state_dict(), quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

if __name__ == "__main__":
    quantize_model()
