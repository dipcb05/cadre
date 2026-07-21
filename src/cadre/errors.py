class CadreError(Exception):
    """Base exception for CADRE."""


class ConfigurationError(CadreError):
    """Raised when a CADRE configuration is invalid."""


class CalibrationError(CadreError):
    """Raised when calibration cannot be completed safely."""


class FeatureExtractionError(CadreError):
    """Raised when feature extraction fails irrecoverably."""


class ProviderError(CadreError):
    """Raised when an external model, retriever, or verifier fails."""


class PolicyError(CadreError):
    """Raised when no valid action can be selected."""


class BudgetExhausted(CadreError):
    """Raised when a nonterminal action is requested without budget."""
