# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)
# from src.translator import Translator
#
#
# def main():
#     # Initialize translator
#     translator = Translator()
#     print("Setting up models...")
#     translator.setup_models()
#     print("Models loaded successfully!")
#
#     print("\nTesting translation...")
#     try:
#         text = "Hello, this is a test!"
#         target_language = "ukr_Cyrl"
#         print(f"Input text: {text}")
#         translated = translator.translate(text, target_language)
#         print(f"Translated text: {translated}")
#     except Exception as e:
#         print(f"Translation test failed: {str(e)}")
#
#
#
#
# if __name__ == "__main__":
#     main()

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')
warnings.filterwarnings('ignore', category=FutureWarning)

from fastapi import FastAPI, HTTPException
from src.translator import Translator
from src.api.schemas import TranslationRequest, TranslationResponse
import uvicorn

app = FastAPI(title="Memo Translator")

translator = None


@app.on_event("startup")
async def startup_event():
    global translator
    try:
        print("Initializing translator...")
        translator = Translator()
        translator.setup_models()
        print("Translator initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize translator: {str(e)}")
        raise


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    try:
        source_language = translator.detect_language(request.text)
        translated_text = translator.translate(
            request.text,
            request.target_language
        )

        return TranslationResponse(
            translated_text=translated_text,
            source_language=source_language,
            target_language=request.target_language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=False)