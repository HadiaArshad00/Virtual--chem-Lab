"""
Virtual Chemistry Lab API - Custom Exceptions
Comprehensive exception hierarchy for the application.
"""

from typing import Optional, Any


class ChemLabException(Exception):
    """Base exception for the Virtual Chemistry Lab API."""

    def __init__(
        self,
        detail: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        extra: Optional[dict] = None,
    ):
        self.detail = detail
        self.status_code = status_code
        self.error_code = error_code
        self.extra = extra or {}
        super().__init__(self.detail)


class ValidationError(ChemLabException):
    """Raised when input validation fails."""

    def __init__(self, detail: str, field: Optional[str] = None):
        self.field = field
        super().__init__(
            detail=detail,
            status_code=422,
            error_code="VALIDATION_ERROR",
            extra={"field": field} if field else {},
        )


class CalculationError(ChemLabException):
    """Raised when a calculation fails."""

    def __init__(self, detail: str, engine: Optional[str] = None):
        self.engine = engine
        super().__init__(
            detail=detail,
            status_code=500,
            error_code="CALCULATION_ERROR",
            extra={"engine": engine} if engine else {},
        )


class EngineNotAvailableError(ChemLabException):
    """Raised when a requested engine is not available."""

    def __init__(self, detail: str, engine: str):
        self.engine = engine
        super().__init__(
            detail=detail,
            status_code=503,
            error_code="ENGINE_UNAVAILABLE",
            extra={"engine": engine},
        )


class ResourceNotFoundError(ChemLabException):
    """Raised when a requested resource is not found."""

    def __init__(self, detail: str, resource_type: Optional[str] = None):
        self.resource_type = resource_type
        super().__init__(
            detail=detail,
            status_code=404,
            error_code="NOT_FOUND",
            extra={"resource_type": resource_type} if resource_type else {},
        )


class AuthenticationError(ChemLabException):
    """Raised when authentication fails."""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            detail=detail,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(ChemLabException):
    """Raised when user is not authorized."""

    def __init__(self, detail: str = "Not authorized"):
        super().__init__(
            detail=detail,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
        )


class RateLimitError(ChemLabException):
    """Raised when rate limit is exceeded."""

    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            detail=detail,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class SMILESValidationError(ValidationError):
    """Raised when SMILES string validation fails."""

    def __init__(self, detail: str, smiles: Optional[str] = None):
        self.smiles = smiles
        super().__init__(detail=detail, field="smiles")


class FileFormatError(ValidationError):
    """Raised when file format is not supported."""

    def __init__(self, detail: str, format: Optional[str] = None):
        self.format = format
        super().__init__(detail=detail, field="format")


class TimeoutError(ChemLabException):
    """Raised when a calculation times out."""

    def __init__(self, detail: str = "Calculation timed out", timeout_seconds: Optional[int] = None):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            detail=detail,
            status_code=504,
            error_code="TIMEOUT",
            extra={"timeout_seconds": timeout_seconds} if timeout_seconds else {},
        )


class DatabaseError(ChemLabException):
    """Raised when a database operation fails."""

    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(
            detail=detail,
            status_code=500,
            error_code="DATABASE_ERROR",
        )
