import time
import montecarlo_engine 

def main():
    # Main example
    S_0 = 100.0
    volatility = 0.2
    K = 100.0
    T = 1.0
    r = 0.05
    OT = montecarlo_engine.OptionType.Call
    N = 50_000_000

    print(f"Initializing Montecarlo simulation with N = {N}...")
    
    start_time = time.perf_counter()
    
    resultado = montecarlo_engine.run_montecarlo(S_0, volatility, K, T, r, OT, N)
    
    end_time = time.perf_counter()
    execution_time_ms = (end_time - start_time) * 1000

    print("\n--- Simulation Results ---")
    print(f"Fair Value:   {resultado.fairValue:.6f}")
    print(f"Confidence interval: [{resultado.CI_lower:.6f}, {resultado.CI_upper:.6f}]")
    print("\n--- Greeks ---")
    print(f"Delta: {resultado.delta:.6f}")
    print(f"Gamma: {resultado.gamma:.6f}")
    print(f"Vega:  {resultado.vega:.6f}")
    print(f"\nAverage price S_T: {resultado.avg_S_T:.6f}")
    
    print(f"\nExecution time in  C++: {execution_time_ms:.2f} miliseconds")

if __name__ == "__main__":
    main()
    