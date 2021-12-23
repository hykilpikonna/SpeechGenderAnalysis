# SpeechGenderAnalysis

## Setup

### MacOS

1. Create virtual environment:

```sh
python3.8 -m venv venv8
source venv8/bin/activate
```

2. Install dependencies:

```sh
pip3 install -r requirements-mac.txt
```

3. Configure `plaidml` to use GPU:

```sh
plaidml-setup
```

4. Configure environment variables:

```sh
export KERAS_BACKEND="plaidml.keras.backend"
export tg_token="Your telegram token here"
```
