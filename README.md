# SpeechGenderAnalysis

## Setup

### MacOS

#### 1. Setup Python:

```sh
python3.8 -m venv venv8
source venv8/bin/activate
pip3 install -r requirements-mac.txt
```

#### 2. Configure `plaidml` to use GPU:

```sh
plaidml-setup
```

#### 3. Configure environment variables in the run script:

```sh
export KERAS_BACKEND="plaidml.keras.backend"
export tg_token="Your telegram token here"
```

### Windows (CUDA)

#### 1. Setup Python

```powershell
python3.9 -m venv venv
.\venv\Scripts\activate
pip install -r requirements-win-cuda.txt
```

#### 2. Install CUDA

* Install NVIDIA Drivers: https://www.nvidia.com/drivers
* Install CUDA **11.2** (for TensorFlow 2.7.0): https://developer.nvidia.com/cuda-toolkit-archive
* Download cuDNN **8.1**: https://developer.nvidia.com/rdp/cudnn-archive
  * Copy folders in `cudnn-11.2-windows-x64-v8.1.1.33\cuda` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`
* Restart IntelliJ IDEA

#### 3. Check Device List

```shell
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```
