"""
MEDEA-NEUMOUSA Configuration
The Talos Destroyer's Settings and Parameters
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """MEDEA-NEUMOUSA application settings with environment variable support"""
    
    # Application Identity
    app_name: str = "MEDEA-NEUMOUSA"
    app_subtitle: str = "The Talos Destroyer - Classical Studies AI Platform"
    app_version: str = "1.0.0"
    app_description: str = "Where ancient languages live again and bronze giants fall"
    debug: bool = False
    environment: str = "development"
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    
    # Security - Medea's Protective Spells
    secret_key: str = "medea-neumousa-secret-change-in-production"
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
    
    # API Keys - Keys to the Digital Underworld
    gemini_api_key: str = "your-gemini-api-key-here"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Paths - The Nine Domains
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    model_dir: Path = data_dir / "models"
    cache_dir: Path = data_dir / "cache"
    logs_dir: Path = base_dir / "logs"
    
    # Model Management - Arsenal of the Muses
    max_model_cache_size: int = 5
    model_timeout: int = 300
    enable_gpu: bool = True
    
    # Classical and Ancient Languages - Voices from the Dead
    classical_languages: List[str] = ["lat", "grc"]  # Core classical
    ancient_languages: List[str] = [
        "he", "arc", "cop", "got", "hit", "sa", "peo", "pal", "syr", "akk", "non", "ang"
    ]
    modern_scholarly: List[str] = ["en", "de", "fr", "it"]  # For scholarly commentary
    supported_languages: List[str] = [
        # Core Classical - The Foundation
        "lat", "grc", 
        # Ancient Near Eastern & Biblical - Sacred Tongues
        "he", "arc", "syc", "akk", "syr",
        # Ancient European - Northern Voices
        "got", "non", "ang", "gmh", "goh",
        # Anatolian & Indo-European - Primordial Languages
        "hit", "luv", "pal", "sa", "av", "peo",
        # Late Antique & Medieval - Transitional Voices
        "cop", "ara", "egy",
        # Modern Scholarly - Contemporary Bridge
        "en", "de", "fr", "it"
    ]
    default_language: str = "lat"
    
    # Processing Parameters - Necromantic Limits
    max_text_length: int = 10000
    batch_size: int = 32
    similarity_threshold: float = 0.7
    clustering_min_samples: int = 2
    translation_confidence_threshold: float = 0.6
    
    # Cache Configuration - Memory of the Dead
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    enable_cache: bool = True
    cache_prefix: str = "medea:"
    
    # Vector Database - Repository of Ancient Souls
    chroma_persist_directory: str = "./data/vectors"
    vector_dimension: int = 768
    collection_name: str = "medea_ancient_texts"
    
    # API Configuration - Endpoints of Power
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    necromancer_endpoint: str = "/necromancer"
    
    # Logging - Chronicle of Deeds
    log_level: str = "INFO"
    log_file: str = "medea.log"
    log_format: str = "%(asctime)s - MEDEA - %(name)s - %(levelname)s - %(message)s"
    
    # Necromancer Specific Settings
    necromancer_model: str = "gemini-2.0-flash"
    necromancer_temperature: float = 0.1
    necromancer_max_tokens: int = 2048
    
    # Performance - Speed of Sorcery
    async_timeout: int = 30
    max_concurrent_requests: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_prefix = "MEDEA_"


# Global settings instance - The Oracle's Voice
settings = Settings()

# Ensure directories exist - Prepare the Sanctuaries
settings.data_dir.mkdir(exist_ok=True)
settings.model_dir.mkdir(exist_ok=True)
settings.cache_dir.mkdir(exist_ok=True)
settings.logs_dir.mkdir(exist_ok=True)

# Language mappings for display
LANGUAGE_NAMES = {
    # Classical
    "lat": "Latin",
    "grc": "Ancient Greek",
    # Ancient Near Eastern
    "he": "Hebrew",
    "arc": "Aramaic", 
    "akk": "Akkadian",
    "syr": "Syriac",
    # Ancient European
    "got": "Gothic",
    "non": "Old Norse",
    "ang": "Old English",
    "gmh": "Middle High German",
    # Anatolian & Indo-European
    "hit": "Hittite",
    "sa": "Sanskrit",
    "av": "Avestan",
    "peo": "Old Persian",
    # Late Antique
    "cop": "Coptic",
    "ara": "Arabic",
    # Modern
    "en": "English",
    "de": "German",
    "fr": "French",
    "it": "Italian"
}
