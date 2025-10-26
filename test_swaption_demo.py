from src.pricing.quantum_pricing import QuantumMonteCarloEngine, QuantumAmplitudeEstimationEngine, HybridPricingEngine
import numpy as np

print('🎯 QUANTUM SWAPTION PRICING DEMO')
print('=' * 40)

# Initialize different quantum pricing engines
monte_carlo_engine = QuantumMonteCarloEngine({})
amplitude_engine = QuantumAmplitudeEstimationEngine({})
hybrid_engine = HybridPricingEngine({})

print('Available pricing engines:')
print('- Quantum Monte Carlo Engine')
print('- Quantum Amplitude Estimation Engine')
print('- Hybrid Pricing Engine')

# Test with Monte Carlo engine
print('\n💰 PRICING SWAPTIONS WITH QUANTUM MONTE CARLO:')
try:
    price = monte_carlo_engine.price_option(100, 105, 1.0, 0.05, 0.2, 'call')
    print(f'Monte Carlo Call Price: ${price:.4f}')
except Exception as e:
    print(f'Error: {e}')

# Test hybrid engine swaption pricing
print('\n💰 PRICING SWAPTIONS WITH HYBRID ENGINE:')
try:
    swaption_price = hybrid_engine.price_swaption_quantum(
        notional=1000000, strike_rate=0.055, time_to_expiry=2.0,
        risk_free_rate=0.03, volatility=0.15, swap_tenor=5.0
    )
    print(f'Hybrid Swaption Price: ${swaption_price:.4f}')
except Exception as e:
    print(f'Error: {e}')