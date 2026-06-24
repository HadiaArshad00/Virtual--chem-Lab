"""
Virtual Chemistry Lab API - Validators
Input validation utilities for chemical data.
"""

import re
from typing import Optional, Tuple
from app.core.utils.exceptions import SMILESValidationError, ValidationError


def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """Validate a SMILES string.

    Args:
        smiles: The SMILES string to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES must be a non-empty string"

    if len(smiles) > 10000:
        return False, "SMILES string too long (max 10000 characters)"

    # Basic SMILES syntax checks
    # Check for balanced parentheses
    open_parens = smiles.count("(")
    close_parens = smiles.count(")")
    if open_parens != close_parens:
        return False, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"

    # Check for balanced brackets
    open_brackets = smiles.count("[")
    close_brackets = smiles.count("]")
    if open_brackets != close_brackets:
        return False, f"Unbalanced brackets: {open_brackets} open, {close_brackets} close"

    # Check for valid characters
    valid_chars = set("ABCDEFGHIKLMNOPRSTUVWXYZabcdefgijklmnoprstuvwxyz0123456789.@#$%&-+=:()[]{}\\/|*~^!?<>")
    invalid_chars = set(smiles) - valid_chars
    if invalid_chars:
        return False, f"Invalid characters in SMILES: {invalid_chars}"

    # Check for ring closure consistency
    ring_digits = re.findall(r"(\d)", smiles)
    for digit in set(ring_digits):
        if ring_digits.count(digit) != 2 and ring_digits.count(digit) != 0:
            return False, f"Ring closure digit {digit} appears {ring_digits.count(digit)} times (must be 0 or 2)"

    return True, ""


def validate_smiles_strict(smiles: str) -> str:
    """Strictly validate a SMILES string and raise exception if invalid.

    Args:
        smiles: The SMILES string to validate.

    Returns:
        The validated SMILES string.

    Raises:
        SMILESValidationError: If SMILES is invalid.
    """
    is_valid, error = validate_smiles(smiles)
    if not is_valid:
        raise SMILESValidationError(detail=error, smiles=smiles)
    return smiles


def validate_molecule_format(format_str: str) -> str:
    """Validate molecule file format.

    Args:
        format_str: The format string to validate.

    Returns:
        Normalized format string.

    Raises:
        ValidationError: If format is not supported.
    """
    supported_formats = {"smiles", "sdf", "mol", "mol2", "pdb", "xyz", "cif", "inchi", "json"}
    format_lower = format_str.lower().strip()

    if format_lower not in supported_formats:
        raise ValidationError(
            detail=f"Unsupported format: {format_str}. Supported: {', '.join(sorted(supported_formats))}",
            field="format",
        )

    return format_lower


def validate_temperature(temperature: float, unit: str = "K") -> float:
    """Validate temperature value.

    Args:
        temperature: The temperature value.
        unit: Temperature unit (K, C, F).

    Returns:
        Temperature in Kelvin.

    Raises:
        ValidationError: If temperature is invalid.
    """
    if temperature is None:
        raise ValidationError(detail="Temperature is required", field="temperature")

    try:
        temp = float(temperature)
    except (TypeError, ValueError):
        raise ValidationError(detail="Temperature must be a number", field="temperature")

    unit = unit.upper()
    if unit == "C":
        temp_kelvin = temp + 273.15
    elif unit == "F":
        temp_kelvin = (temp - 32) * 5 / 9 + 273.15
    elif unit == "K":
        temp_kelvin = temp
    else:
        raise ValidationError(detail=f"Invalid temperature unit: {unit}", field="temperature_unit")

    if temp_kelvin < 0:
        raise ValidationError(detail="Temperature cannot be below absolute zero (0 K)", field="temperature")

    if temp_kelvin > 10000:
        raise ValidationError(detail="Temperature too high (max 10000 K)", field="temperature")

    return temp_kelvin


def validate_pressure(pressure: float, unit: str = "atm") -> float:
    """Validate pressure value.

    Args:
        pressure: The pressure value.
        unit: Pressure unit (atm, Pa, bar, mmHg, psi).

    Returns:
        Pressure in atmospheres.

    Raises:
        ValidationError: If pressure is invalid.
    """
    if pressure is None:
        raise ValidationError(detail="Pressure is required", field="pressure")

    try:
        p = float(pressure)
    except (TypeError, ValueError):
        raise ValidationError(detail="Pressure must be a number", field="pressure")

    if p < 0:
        raise ValidationError(detail="Pressure cannot be negative", field="pressure")

    # Convert to atm
    unit = unit.lower()
    conversion = {
        "atm": 1.0,
        "pa": 9.86923e-6,
        "bar": 0.986923,
        "mmhg": 1 / 760,
        "torr": 1 / 760,
        "psi": 1 / 14.696,
    }

    if unit not in conversion:
        raise ValidationError(detail=f"Invalid pressure unit: {unit}", field="pressure_unit")

    return p * conversion[unit]


def validate_api_key(api_key: str) -> bool:
    """Validate API key format.

    Args:
        api_key: The API key to validate.

    Returns:
        True if valid format.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    if len(api_key) < 32:
        return False
    # API keys should be alphanumeric with some special chars
    if not re.match(r"^[A-Za-z0-9_-]+$", api_key):
        return False
    return True
