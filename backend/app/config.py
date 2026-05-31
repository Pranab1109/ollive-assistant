import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Gemini API Configuration
    GEMINI_API_KEY: str = Field(default="", validation_alias="GEMINI_API_KEY")
    
    # Ollama Local Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    OSS_MODEL_NAME: str = Field(default="llama3.2", validation_alias="OSS_MODEL_NAME")
    
    # Public Model Server Configuration (to be set after deployment)
    HF_SPACE_MODEL_URL: str = Field(default="", validation_alias="HF_SPACE_MODEL_URL")
    HF_TOKEN: str = Field(default="", validation_alias="HF_TOKEN")
    
    # Server configuration
    PORT: int = Field(default=7860, validation_alias="PORT")
    
    # Enable reading from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
