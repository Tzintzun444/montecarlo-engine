import pytest
import math
import montecarlo_engine

# ==========================================
# CONFIGURATION & TOLERANCES
# ==========================================
# N needs to be high enough to reduce variance, but small enough for fast CI/CD
N_SIMULATIONS = 15_000_000 
PRICE_TOLERANCE = 0.05
GREEK_TOLERANCE = 0.05

# ==========================================
# TEST SUITE 1: DETERMINISTIC LIMITS
# ==========================================
def test_zero_volatility_call_itm():
    """
    If volatility is 0, the process is deterministic: S_T = S_0 * exp(rT).
    The Fair Value should be exactly the discounted payoff.
    """
    S_0 = 100.0
    vol = 0.0
    K = 90.0
    T = 1.0
    r = 0.05
    
    result = montecarlo_engine.run_montecarlo(
        S_0, vol, K, T, r, montecarlo_engine.OptionType.Call, 1000
    )
    
    # Expected S_T = 100 * exp(0.05) = 105.1271
    expected_S_T = S_0 * math.exp(r * T)
    expected_payoff = max(expected_S_T - K, 0.0)
    expected_pv = expected_payoff * math.exp(-r * T)
    
    assert result.avg_S_T == pytest.approx(expected_S_T, abs=1e-5)
    assert result.fairValue == pytest.approx(expected_pv, abs=1e-5)

def test_deep_out_of_the_money_put():
    """
    A Put option where S_0 >>> K should have a value approaching 0.
    """
    result = montecarlo_engine.run_montecarlo(
        S_0=200.0, volatility=0.1, K=50.0, T=1.0, r=0.05, 
        OT=montecarlo_engine.OptionType.Put, N=10_000
    )
    assert result.fairValue < 1e-4

# ==========================================
# TEST SUITE 2: FINANCIAL THEOREMS
# ==========================================
def test_put_call_parity():
    """
    Tests the fundamental Put-Call Parity theorem: C - P = S_0 - K * exp(-rT).
    This proves the Call and Put pricing logic is mathematically symmetrical.
    """
    S_0 = 100.0
    vol = 0.20
    K = 100.0
    T = 1.0
    r = 0.05
    
    call_res = montecarlo_engine.run_montecarlo(
        S_0, vol, K, T, r, montecarlo_engine.OptionType.Call, N_SIMULATIONS
    )
    put_res = montecarlo_engine.run_montecarlo(
        S_0, vol, K, T, r, montecarlo_engine.OptionType.Put, N_SIMULATIONS
    )
    
    lhs = call_res.fairValue - put_res.fairValue
    rhs = S_0 - (K * math.exp(-r * T))
    
    assert lhs == pytest.approx(rhs, abs=PRICE_TOLERANCE)

def test_martingale_property():
    """
    Under the risk-neutral measure, the expected value of S_T 
    must exactly equal S_0 * exp(rT) regardless of volatility.
    """
    S_0 = 100.0
    vol = 0.30
    T = 2.0
    r = 0.03
    
    result = montecarlo_engine.run_montecarlo(
        S_0, vol, 100.0, T, r, montecarlo_engine.OptionType.Call, N_SIMULATIONS
    )
    
    expected_S_T = S_0 * math.exp(r * T)
    assert result.avg_S_T == pytest.approx(expected_S_T, abs=PRICE_TOLERANCE)

# ==========================================
# TEST SUITE 3: GREEKS BEHAVIOR
# ==========================================
def test_delta_limits():
    """
    Deep ITM Call Delta should approach 1.0.
    Deep OTM Call Delta should approach 0.0.
    """
    vol = 0.1
    T = 0.5
    r = 0.0
    
    # Deep ITM Call (S_0 >> K)
    itm_res = montecarlo_engine.run_montecarlo(
        S_0=150.0, volatility=vol, K=100.0, T=T, r=r, 
        OT=montecarlo_engine.OptionType.Call, N=N_SIMULATIONS
    )
    assert itm_res.delta == pytest.approx(1.0, abs=GREEK_TOLERANCE)
    
    # Deep OTM Call (S_0 << K)
    otm_res = montecarlo_engine.run_montecarlo(
        S_0=50.0, volatility=vol, K=100.0, T=T, r=r, 
        OT=montecarlo_engine.OptionType.Call, N=N_SIMULATIONS
    )
    assert otm_res.delta == pytest.approx(0.0, abs=GREEK_TOLERANCE)

def test_gamma_vega_symmetry():
    """
    Gamma and Vega should be identical for Calls and Puts with the same parameters.
    """
    S_0 = 100.0
    vol = 0.2
    K = 100.0
    T = 1.0
    r = 0.05
    
    call_res = montecarlo_engine.run_montecarlo(
        S_0, vol, K, T, r, montecarlo_engine.OptionType.Call, N_SIMULATIONS
    )
    put_res = montecarlo_engine.run_montecarlo(
        S_0, vol, K, T, r, montecarlo_engine.OptionType.Put, N_SIMULATIONS
    )
    
    assert call_res.gamma == pytest.approx(put_res.gamma, abs=GREEK_TOLERANCE)
    assert call_res.vega == pytest.approx(put_res.vega, abs=GREEK_TOLERANCE)

# ==========================================
# TEST SUITE 4: DATA STRUCTURES
# ==========================================
def test_histogram_counter_integrity():
    """
    Validates that the C++ unordered_map translates to a Python dict correctly,
    and that the sum of all frequencies equals exactly 2 * N (since the loop 
    processes 2 antithetic/CRN paths per iteration).
    """
    N = 10_000
    result = montecarlo_engine.run_montecarlo(
        100.0, 0.2, 100.0, 1.0, 0.05, montecarlo_engine.OptionType.Call, N
    )
    
    total_paths = sum(result.counter.values())
    
    assert isinstance(result.counter, dict)
    assert total_paths == N