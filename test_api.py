"""
Virtual Chemistry Lab API - API Tests
"""

import pytest


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data


def test_list_calculations(client):
    """Test listing calculation types."""
    response = client.get("/api/v1/calculations")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_calculate_descriptors(client):
    """Test descriptor calculation."""
    response = client.post(
        "/api/v1/calculations/descriptors",
        params={"smiles": "CCO"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "molecular_weight" in data["data"]


def test_calculate_fingerprint(client):
    """Test fingerprint calculation."""
    response = client.post(
        "/api/v1/calculations/fingerprint",
        params={"smiles": "CCO", "fingerprint_type": "morgan"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_predict_yield(client):
    """Test yield prediction."""
    response = client.post(
        "/api/v1/calculations/yield",
        json={
            "reactants_smiles": ["CCO", "O=C(O)C"],
            "conditions": {"temperature": 25, "time_hours": 2, "solvent": "DMF"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "predicted_yield" in data["data"]


def test_predict_pka(client):
    """Test pKa prediction."""
    response = client.post(
        "/api/v1/calculations/pka",
        json={"smiles": "CC(=O)O"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_predict_logp(client):
    """Test LogP prediction."""
    response = client.post(
        "/api/v1/calculations/logp",
        json={"smiles": "CCO"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_recommend_solvent(client):
    """Test solvent recommendation."""
    response = client.post(
        "/api/v1/calculations/solvent",
        json={"reaction_type": "coupling", "top_n": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_kinetics_arrhenius(client):
    """Test Arrhenius calculation."""
    response = client.post(
        "/api/v1/calculations/kinetics",
        json={
            "calculation_type": "arrhenius",
            "A": 1e10,
            "Ea": 50000,
            "temperatures": [298.15, 308.15, 318.15]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_spectra_ir(client):
    """Test IR spectra calculation."""
    response = client.post(
        "/api/v1/calculations/spectra",
        json={"smiles": "CCO", "spectra_type": "ir"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
