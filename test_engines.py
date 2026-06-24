"""
Virtual Chemistry Lab API - Engine Tests
"""

import pytest
from app.core.engines.rdkit_engine import RDKitEngine


@pytest.fixture
def rdkit_engine():
    return RDKitEngine()


@pytest.mark.asyncio
async def test_rdkit_validate_smiles(rdkit_engine):
    """Test SMILES validation."""
    result = await rdkit_engine.calculate({
        "method": "smiles_validation",
        "smiles": "CCO"
    })
    assert result.success is True
    assert result.data["valid"] is True


@pytest.mark.asyncio
async def test_rdkit_molecular_weight(rdkit_engine):
    """Test molecular weight calculation."""
    result = await rdkit_engine.calculate({
        "method": "molecular_weight",
        "smiles": "CCO"
    })
    assert result.success is True
    assert result.data["molecular_weight"] > 0


@pytest.mark.asyncio
async def test_rdkit_descriptors(rdkit_engine):
    """Test descriptor calculation."""
    result = await rdkit_engine.calculate({
        "method": "descriptors",
        "smiles": "CCO"
    })
    assert result.success is True
    assert "logp" in result.data
    assert "tpsa" in result.data


@pytest.mark.asyncio
async def test_rdkit_fingerprint(rdkit_engine):
    """Test fingerprint generation."""
    result = await rdkit_engine.calculate({
        "method": "morgan_fingerprint",
        "smiles": "CCO",
        "n_bits": 2048
    })
    assert result.success is True
    assert result.data["n_bits"] == 2048
