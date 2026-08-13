import pytest
import numpy as np

from core.metrics.loss_diagnostics import InformationLossDiagnostics

@pytest.fixture
def dummy_vectors():
    return np.random.rand(50, 128)

def test_mantle_test_identity(dummy_vectors):
    # Comparing the same matrix should yield 1.0 correlation
    corr = InformationLossDiagnostics.mantle_test(dummy_vectors, dummy_vectors)
    assert np.isclose(corr, 1.0), "Identity mantle test should return 1.0"

def test_mantle_test_degradation(dummy_vectors):
    # Add noise to simulate degradation
    noisy_vectors = dummy_vectors + np.random.normal(0, 0.5, dummy_vectors.shape)
    
    corr = InformationLossDiagnostics.mantle_test(dummy_vectors, noisy_vectors)
    assert corr < 1.0, "Degradation should lower the correlation"

def test_reconstruction_stress(dummy_vectors):
    # Perfect reconstruction
    mse_perfect = InformationLossDiagnostics.reconstruction_stress(dummy_vectors, dummy_vectors)
    assert mse_perfect == 0.0, "Perfect reconstruction should have 0 MSE"
    
    # Degraded reconstruction
    noisy_vectors = dummy_vectors + np.random.normal(0, 0.5, dummy_vectors.shape)
    mse_noisy = InformationLossDiagnostics.reconstruction_stress(dummy_vectors, noisy_vectors)
    assert mse_noisy > 0.0, "Degraded reconstruction should have > 0 MSE"

def test_trustworthiness(dummy_vectors):
    # Identity preservation
    score = InformationLossDiagnostics.trustworthiness(dummy_vectors, dummy_vectors, n_neighbors=5)
    assert score == 1.0, "Trustworthiness of identity should be 1.0"
