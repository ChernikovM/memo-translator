from pydantic import BaseModel, Field

class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language code (e.g., 'ukr_Cyrl')")

class TranslationResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str