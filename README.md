# SDFI_ml

Machine learning framework developed and maintained by **KDS** for performing **semantic segmentation** on **multichannel images**.

---

## Installation

### Conda version

Use **conda** or **mamba** (Miniforge includes conda; mamba is optional). Clone all four shared-env repos as siblings (`ML_Production`, `ML_geo_production`, `ML_sdfi_fastai2`, `multi_channel_dataset_creation`), then run the steps below **from this repository root**. The same files and commands exist in each repo and produce the same `ML_sdfi` environment.

```sh
conda env create --file environment.yml   # once
conda activate ML_sdfi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # Linux

bash install_pytorch.sh
pip install --pre --no-build-isolation -r requirements_pip.txt
bash install_local_repos.sh
pip install -r requirements_extra.txt
```

`install_pytorch.sh` auto-selects the PyTorch CUDA build (nightly cu128 for Blackwell / sm_12.0, stable cu124 for other NVIDIA GPUs). CUDA is required. Override with e.g. `PYTORCH_CUDA=cu121 bash install_pytorch.sh` (see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)).

**Verify CUDA support:**

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You should see `CUDA available: True` and your GPU name.

**Windows:** After the steps above, run once: `pip install --force-reinstall pillow rasterio` so PIL and rasterio use pip's Windows wheels.

### Docker version

Build and run the repo's Docker image:

```sh
docker build -t ml_sdfi_fastai2-dev-env:latest .

docker run --gpus all --shm-size=80g -it \
  -v "$(realpath ..):/projects" \
  -v /mnt/T/mnt:/mnt/T/mnt \
  -w /projects/ML_sdfi_fastai2 \
  ml_sdfi_fastai2-dev-env:latest /bin/bash
```

Or pull the shared prebuilt image and run with this repo as working directory:

```bash
docker pull rasmuspjohansson/kds_cuda_pytorch:latest

docker run --gpus all --shm-size=80g -it \
  -v "$(realpath ..):/projects" \
  -v /mnt/T/mnt:/mnt/T/mnt \
  -w /projects/ML_sdfi_fastai2 \
  rasmuspjohansson/kds_cuda_pytorch:latest /bin/bash
```

(Adjust volume paths as needed. To have all four shared-env repos inside the container, run once from ML_Production: `sh install_local_repos.sh && pip install -r requirements_extra.txt`.)

---

## Verify that everything works

Manual check — CUDA smoke test:

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You should see `CUDA available: True` and your GPU name.

Automated verification (CUDA check — fails if CUDA unavailable; trains each `configs/example_configs/test*.ini` config; writes `verification.log`):

```sh
python verify_functionality.py
python check_logs.py
```

This branch has five `test*.ini` configs (ResNet34, SegFormer, ConvNeXt V2, Swin UPerNet, etc.). Set `VERIFY_TRAIN_TIMEOUT=1800` for longer training runs if needed.

---

## Usage

### Training

Configuration file:  
`configs/example_configs/train_example_dataset.ini`

Run:
```sh
python src/ML_sdfi_fastai2/train.py --config configs/example_configs/train_example_dataset.ini
```

---

### Inference

Configuration file:  
`configs/example_configs/infer_example_dataset.ini`

Run:
```sh
python src/ML_sdfi_fastai2/infer.py --config configs/example_configs/infer_example_dataset.ini
```

---

### Reporting

Configuration file:  
`configs/example_configs/report_example_dataset.ini`

Run:
```sh
In order to analyse how well the model does on a dataset we can create a .pdf report with loss plots , confusion matrixes and example iamges and predictions

cd ../multi_channel_dataset_creation
python src/multi_channel_dataset_creation/geopackage_to_label_v2.py --geopackage example_dataset/labels/example_dataset_buildings.gpkg --input_folder example_dataset/data/splitted/rgb/ --output_folder example_dataset/buildings/splitted_buildings/ --background_value 0 --value_used_for_all_polygons 1
cd ../ML_sdfi_fastai2
python src/ML_sdfi_fastai2/report.py --config configs/example_configs/report_example_dataset.ini

If you get confusing errors when creating report it is often casued by a faulty .csv file 
check ../logs_and_models/example_dataset_iter_1/iter_1/logs/iter_1.csv 
This file should look like this

epoch,train_loss,valid_loss,valid_accuracy,time,lr_0,lr_1,lr_2
0,2.671426296234131,2.261967182159424,0.15685869753360748,00:07,8e-05,8e-05,8e-05
1,2.4981231689453125,1.2677741050720215,0.6439558863639832,00:06
...
...
9,3.6438961029052734,1.2790676355361938,0.6442530751228333,00:06,8.64737438191444e-05,8.64737438191444e-05,8.64737438191444e-05

```

---

### Downloading and uploading models (Hugging Face)

Trained `.pth` models can be downloaded from or uploaded to [Hugging Face Hub](https://huggingface.co/). The default repo used by the scripts is [rasmuspjohansson/KDS_buildings](https://huggingface.co/rasmuspjohansson/KDS_buildings). The dependency `huggingface_hub` is included in the environment; ensure it is installed (`pip install huggingface_hub` if needed).

#### Download models from Hugging Face

Download one or all `.pth` model files from the repo into a local directory:

```sh
# Download all .pth files from the repo into a folder
python src/ML_sdfi_fastai2/download_models_from_huggingface.py \
  --output_dir /mnt/T/mnt/logs_and_models/bygningsudpegning

# Download a single model file
python src/ML_sdfi_fastai2/download_models_from_huggingface.py \
  --output_dir ./models \
  --model_file andringsudpegning_1km2benchmark_iter_73.pth
```

Options:
- `--repo_id`: Hugging Face repo (default: `rasmuspjohansson/KDS_buildings`)
- `--output_dir`: Local directory where files are saved (required)
- `--model_file`: If set, only this filename is downloaded; otherwise all `.pth` files in the repo are downloaded
- `--token_file`: Path to a file containing your Hugging Face token (for private repos). Default: `../laz-superpoint_transformer/hftoken_write.txt` relative to the project root

To use a downloaded model for inference, point your inference config’s `model_to_load` to the downloaded file, e.g. `.../bygningsudpegning/andringsudpegning_1km2benchmark_iter_73.pth`.

#### Upload models to Hugging Face

Upload models that were trained using `train.py` and whose configs live in a directory of `train_*.ini` files. The script reads each config’s `experiment_root` and `job_name`, finds the corresponding `{experiment_root}/{job_name}/models/{job_name}.pth`, and uploads it to the Hub:

```sh
# Upload all models from configs in the default config directory
python src/ML_sdfi_fastai2/upload_models_to_huggingface.py \
  --config_dir /mnt/T/mnt/config_files/bygnings_udpegning/2026_production/

# Dry run: only print what would be uploaded
python src/ML_sdfi_fastai2/upload_models_to_huggingface.py \
  --config_dir /mnt/T/mnt/config_files/bygnings_udpegning/2026_production/ \
  --dry_run
```

Options:
- `--config_dir`: Directory containing `train_*.ini` config files (default: `/mnt/T/mnt/config_files/bygnings_udpegning/2026_production`)
- `--token_file`: Path to a file containing your Hugging Face **write** token. Default: `../laz-superpoint_transformer/hftoken_write.txt` relative to the project root
- `--repo_id`: Target Hugging Face repo (default: `rasmuspjohansson/KDS_buildings`)
- `--dry_run`: List configs and model paths that would be uploaded without uploading

Only configs for which the corresponding `.pth` file exists are uploaded; others are skipped with a message.

---

## Example Dataset

All example configuration files are compatible with the example dataset available at:  
👉 [https://github.com/SDFIdk/multi_channel_dataset_creation](https://github.com/SDFIdk/multi_channel_dataset_creation)




