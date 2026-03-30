# DINO-EEG: Leveraging Object Detection Models for Seizure Event Detection and Type Classification

## Acknowledgment
This implementation is bulit upon [DINO](https://github.com/IDEA-Research/DINO/)

## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 22.04
* Python: 3.11.8
* CUDA: 12.4
* PyTorch: 2.2.0 (The lower versions of Torch can cause some bugs.)
* torchvision: 0.17.0

We use the environment almost the same to DINO to run DINO-EEG.

1. Install Pytorch and torchvision

    Follow the instruction on https://pytorch.org/get-started/locally/.
    ```sh
    # an example:
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
   ```

2. Install other needed packages
    ```sh
    pip install -r requirements.txt
    ```

3. Compiling CUDA operators
    ```sh
    cd models/dino/ops
    python setup.py build install
    # unit test (should see all checking is True)
    python test.py
    cd ../../..
    ```

## Data Preparation

### TUSZ v2.0.3 Dataset

1. Download [TUSZ v2.0.3](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tusz) dataset

2. Preprocessing
    ```sh
    cd datasets
    python tusz/preprocess_wo_slice.py --raw_edf_dir Raw_TUSZ_Dir/train/ --save_dir Preprocess_Data_Dir/train/ && python tusz/preprocess_wo_slice.py --raw_edf_dir Raw_TUSZ_Dir/dev/ --save_dir Preprocess_Data_Dir/dev/ && python tusz/preprocess_wo_slice.py --raw_edf_dir Raw_TUSZ_Dir/eval/ --save_dir Preprocess_Data_Dir/eval/
    ```

3. STFT transform
    ```sh
    python common/transform.py --data_dir Preprocess_Data_Dir/train/ --save_dir STFT_Data_Dir/train/ --scale --crop && python common/transform.py --data_dir Preprocess_Data_Dir/dev/ --save_dir STFT_Data_Dir/dev/ --scale --crop && python common/transform.py --data_dir Preprocess_Data_Dir/eval/ --save_dir STFT_Data_Dir/eval/ --scale --crop
    ```

4. Make Binary Classification Labels
    ```sh
    python common/change_slice_label_only.py --data_dir STFT_Data_Dir/train/ --save_dir Binary_Label_Dir/train/ && python common/change_slice_label_only.py --data_dir STFT_Data_Dir/dev/ --save_dir Binary_Label_Dir/dev/ && python common/change_slice_label_only.py --data_dir STFT_Data_Dir/eval/ --save_dir Binary_Label_Dir/eval/
    ```

5. Make .txt file
    ```sh
    python tusz/filter_and_sort.py STFT_Data_Dir/..(The parent folder of STFT_Data_Dir)
    ```
    Copy the generated .txt files to TXT_file_Dir.

### CHB-MIT Dataset

1. Download [CHB-MIT](https://physionet.org/content/chbmit/1.0.0/) dataset

2. Generate labels from .txt files
    ```sh
    cd datasets
    python chbmit/make_label.py Raw_CHBMIT_Dir Label_Save_Dir
    ```

3. Preprocessing
    ```sh
    python chbmit/preprocess.py --raw_edf_dir Raw_CHBMIT_Dir --preprocessed_label_dir Label_Save_Dir --save_dir Preprocess_Data_Dir
    ```

4. STFT transform
    ```sh
    python common/transform.py --data_dir Preprocess_Data_Dir/train/ --save_dir STFT_Data_Dir/train/ --scale --crop && python common/transform.py --data_dir Preprocess_Data_Dir/dev/ --save_dir STFT_Data_Dir/dev/ --scale --crop && python common/transform.py --data_dir Preprocess_Data_Dir/eval/ --save_dir STFT_Data_Dir/eval/ --scale --crop
    ```

5. Make .txt file
    ```sh
    python tusz/filter_and_sort.py STFT_Data_Dir/..(The parent folder of STFT_Data_Dir)
    ```
    Copy the generated .txt files to TXT_file_Dir.

## Run

### Training

1. Load pretrained checkpoint to fine-tune
    Modify scripts/DINO_train_swin_tusz.sh, add --pretrain_model_path /path/to/your/checkpoints/xxx.
    ```sh
    bash scripts/DINO_train_swin_tusz.sh cuda:0
    ```

2.  Train from scratch
    Delete --pretrain_model_path in scripts/DINO_train_swin_tusz.sh.
    ```sh
    bash scripts/DINO_train_swin_tusz.sh cuda:0
    ```

3. Binary detection
    Modify scripts/DINO_train_swin_tusz.sh, add --tusz_label_dir /path/to/your/tusz/Binary_Label_Dir.
    ```sh
    bash scripts/DINO_train_swin_tusz.sh cuda:0
    ```

### Evaluation

1. Evaluation
    ```sh
    bash scripts/DINO_eval_tusz.sh /path/to/your/logs/DINO/TUSZ/foo/checkpoint_best_regular.pth cuda:0
    ```

## Reference
https://github.com/IDEA-Research/DINO