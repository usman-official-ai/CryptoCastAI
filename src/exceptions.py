"""Custom exceptions for CryptoCastAI."""


class CryptoCastAIError(Exception):
    """Base exception."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class DataFetchError(CryptoCastAIError):
    """Data fetching failed."""
    pass


class DataValidationError(CryptoCastAIError):
    """Data validation failed."""
    pass


class ModelTrainingError(CryptoCastAIError):
    """Model training failed."""
    pass


class ModelLoadError(CryptoCastAIError):
    """Model loading failed."""
    pass


class PredictionError(CryptoCastAIError):
    """Prediction failed."""
    pass