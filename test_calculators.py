"""
Virtual Chemistry Lab API - Calculator Tests
"""

import pytest
from app.core.calculators.kinetics import KineticsCalculator
from app.core.calculators.spectra import SpectraCalculator


@pytest.fixture
def kinetics_calc():
    return KineticsCalculator()


@pytest.fixture
def spectra_calc():
    return SpectraCalculator()


@pytest.mark.asyncio
async def test_arrhenius_rate(kinetics_calc):
    """Test Arrhenius equation calculation."""
    result = await kinetics_calc.arrhenius_rate(
        A=1e10,
        Ea=50000,
        temperatures=[298.15, 308.15]
    )
    assert result.success is True
    assert len(result.data["rate_constants"]) == 2


@pytest.mark.asyncio
async def test_eyring_rate(kinetics_calc):
    """Test Eyring equation calculation."""
    result = await kinetics_calc.eyring_rate(
        delta_G=80000,
        temperatures=[298.15]
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_ir_spectra(spectra_calc):
    """Test IR spectra prediction."""
    result = await spectra_calc.calculate_ir("CCO")
    assert result.success is True
    assert "peaks" in result.data


@pytest.mark.asyncio
async def test_nmr_spectra(spectra_calc):
    """Test NMR spectra prediction."""
    result = await spectra_calc.calculate_nmr("CCO", nuclei="1H")
    assert result.success is True
    assert "shifts" in result.data
