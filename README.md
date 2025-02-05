How to run:

pip install -r requirements.txt

python main.py


.env:
# Model checkpoint from Hugging Face (available options):
# facebook/nllb-200-distilled-600M
# facebook/nllb-200-1.3B
# facebook/nllb-200-3.3B
# facebook/nllb-200-distilled-1.3B
MODEL_CHECKPOINT=facebook/nllb-200-3.3B

# Path to language detection model
LANGUAGE_MODEL_PATH=./pretrained/lid218e.bin

# Directory for storing downloaded models
LOCAL_MODELS_DIR=models

# Maximum text length for translation
MAX_LENGTH=500

LANG_MODEL_URL='https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin'
