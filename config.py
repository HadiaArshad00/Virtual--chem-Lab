"""
Virtual Chemistry Lab API - Configuration
Pydantic Settings with environment variable loading and sensible defaults.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://chem_user:chem_pass@localhost/chem_lab",
        description="PostgreSQL connection string",
    )

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )

    # Celery
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL",
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL",
    )

    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production-virtual-chem-lab-2026",
        description="Secret key for JWT token generation",
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="JWT token expiration time in minutes",
    )

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=100,
        description="Maximum requests per minute per user",
    )

    # Engine Configuration
    DEFAULT_ENGINE: str = Field(
        default="rdkit",
        description="Default calculation engine",
    )
    PSI4_AVAILABLE: bool = Field(
        default=False,
        description="Whether Psi4 quantum chemistry engine is available",
    )
    VINA_AVAILABLE: bool = Field(
        default=False,
        description="Whether AutoDock Vina is available",
    )
    OPENBABEL_AVAILABLE: bool = Field(
        default=True,
        description="Whether Open Babel is available",
    )

    # ML Model Paths
    YIELD_MODEL_PATH: str = Field(
        default="app/core/ml/models/yield_predictor.pkl",
        description="Path to yield prediction model",
    )
    PKA_MODEL_PATH: str = Field(
        default="app/core/ml/models/pka_predictor.pkl",
        description="Path to pKa prediction model",
    )
    LOGP_MODEL_PATH: str = Field(
        default="app/core/ml/models/logp_predictor.pkl",
        description="Path to LogP prediction model",
    )
    SOLVENT_DB_PATH: str = Field(
        default="app/core/ml/models/solvent_database.json",
        description="Path to solvent database",
    )

    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins",
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level",
    )
    LOG_FORMAT: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Loguru format string",
    )

    # Calculation Defaults
    DEFAULT_BASIS_SET: str = Field(
        default="def2-SVP",
        description="Default basis set for DFT calculations",
    )
    DEFAULT_FUNCTIONAL: str = Field(
        default="PBE0",
        description="Default DFT functional",
    )
    MAX_ATOMS_DFT: int = Field(
        default=50,
        description="Maximum number of atoms for DFT calculations",
    )
    MD_TIMESTEP: float = Field(
        default=1.0,
        description="MD timestep in femtoseconds",
    )
    MD_STEPS: int = Field(
        default=10000,
        description="Default number of MD steps",
    )

    # Docking
    VINA_EXECUTABLE: str = Field(
        default="vina",
        description="AutoDock Vina executable path",
    )
    VINA_NUM_POSES: int = Field(
        default=10,
        description="Number of poses to generate",
    )

    # Batch Processing
    MAX_BATCH_SIZE: int = Field(
        default=100,
        description="Maximum experiments in a batch job",
    )

    # File Storage
    UPLOAD_DIR: str = Field(
        default="/tmp/chem_lab/uploads",
        description="Directory for uploaded files",
    )
    EXPORT_DIR: str = Field(
        default="/tmp/chem_lab/exports",
        description="Directory for exported files",
    )
    MAX_UPLOAD_SIZE: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        description="Maximum upload file size in bytes",
    )

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
