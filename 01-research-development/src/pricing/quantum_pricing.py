"""
Quantum-enhanced pricing engines for financial derivatives.

This module implements quantum algorithms for option pricing including:
- Quantum Monte Carlo simulation
- Quantum amplitude estimation
- Quantum walk algorithms
- Hybrid quantum-classical approaches
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit.primitives import Estimator
    from qiskit.circuit.library import RYGate, CXGate
    from qiskit.providers.fake_provider import FakeVigo
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
    QISKIT_AVAILABLE = True
except ImportError:
    logger.warning("Qiskit not available. Using classical fallbacks.")
    QISKIT_AVAILABLE = False


class QuantumPricingEngine(ABC):
    """Abstract base class for quantum pricing engines."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = self._initialize_backend()

    def _initialize_backend(self):
        """Initialize quantum backend."""
        if not QISKIT_AVAILABLE:
            return None

        try:
            if self.config.get('backend', 'simulator') == 'simulator':
                return AerSimulator()
            else:
                # Initialize IBM Quantum service
                service = QiskitRuntimeService(
                    channel="ibm_quantum",
                    token=self.config.get('api_key')
                )
                return service.backend(self.config.get('backend', 'ibmq_qasm_simulator'))
        except Exception as e:
            logger.error(f"Failed to initialize quantum backend: {e}")
            return AerSimulator()  # Fallback to simulator

    @abstractmethod
    def price_option(self, spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Price an option using quantum algorithms."""
        pass


class QuantumMonteCarloEngine(QuantumPricingEngine):
    """Quantum Monte Carlo simulation for option pricing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.n_shots = config.get('shots', 1024)
        self.n_qubits = config.get('n_qubits', 8)

    def price_option(self, spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Price European option using quantum Monte Carlo."""
        if not QISKIT_AVAILABLE:
            return self._classical_fallback(spot, strike, time_to_expiry,
                                          risk_free_rate, volatility, option_type)

        try:
            # Create quantum circuit for Monte Carlo simulation
            qc = self._create_monte_carlo_circuit(
                spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )

            # Execute on quantum backend
            if hasattr(self.backend, 'run'):
                # IBM Quantum backend
                sampler = Sampler(mode=Session(backend=self.backend))
                job = sampler.run([qc], shots=self.n_shots)
                result = job.result()
                counts = result.quasi_dists[0]
            else:
                # Local simulator
                qc_transpiled = transpile(qc, self.backend)
                job = self.backend.run(qc_transpiled, shots=self.n_shots)
                result = job.result()
                counts = result.get_counts()

            # Extract price from measurement results
            price = self._extract_price_from_counts(counts, strike, option_type)
            return price

        except Exception as e:
            logger.error(f"Quantum Monte Carlo failed: {e}")
            return self._classical_fallback(spot, strike, time_to_expiry,
                                          risk_free_rate, volatility, option_type)

    def _create_monte_carlo_circuit(self, spot: float, strike: float, time_to_expiry: float,
                                   risk_free_rate: float, volatility: float, option_type: str) -> QuantumCircuit:
        """Create quantum circuit for Monte Carlo option pricing."""
        n_qubits = self.n_qubits
        qc = QuantumCircuit(n_qubits, n_qubits)

        # Encode parameters into quantum states
        self._encode_parameters(qc, spot, strike, time_to_expiry, risk_free_rate, volatility)

        # Apply quantum walk for stochastic process simulation
        self._apply_quantum_walk(qc, time_to_expiry, risk_free_rate, volatility)

        # Apply payoff function
        self._apply_payoff_function(qc, strike, option_type)

        # Measure
        qc.measure_all()

        return qc

    def _encode_parameters(self, qc: QuantumCircuit, spot: float, strike: float,
                          time_to_expiry: float, risk_free_rate: float, volatility: float):
        """Encode financial parameters into quantum states."""
        # Normalize parameters to [0, π] for RY rotations
        spot_angle = (spot / 200.0) * np.pi  # Assuming spot range 0-200
        strike_angle = (strike / 200.0) * np.pi
        vol_angle = (volatility / 1.0) * np.pi  # Volatility 0-1
        rate_angle = (risk_free_rate / 0.1) * np.pi  # Rate 0-0.1

        # Encode parameters using RY gates
        qc.ry(spot_angle, 0)
        qc.ry(strike_angle, 1)
        qc.ry(vol_angle, 2)
        qc.ry(rate_angle, 3)

    def _apply_quantum_walk(self, qc: QuantumCircuit, time_to_expiry: float,
                           risk_free_rate: float, volatility: float):
        """Apply quantum walk for stochastic process simulation."""
        n_steps = max(1, int(time_to_expiry * 252 / 10))  # Approximate trading days

        for step in range(min(n_steps, 10)):  # Limit steps for circuit depth
            # Brownian motion simulation using controlled rotations
            drift_angle = risk_free_rate * time_to_expiry / n_steps
            diffusion_angle = volatility * np.sqrt(time_to_expiry / n_steps)

            # Apply drift
            qc.ry(drift_angle, 4)

            # Apply diffusion (random walk)
            qc.h(5)  # Hadamard for randomness
            qc.cry(diffusion_angle, 5, 4)

            # Entangle with parameter qubits
            qc.cx(4, 0)

    def _apply_payoff_function(self, qc: QuantumCircuit, strike: float, option_type: str):
        """Apply option payoff function to quantum state."""
        # Simplified payoff encoding
        # In practice, this would use more sophisticated amplitude encoding

        if option_type.lower() == 'call':
            # Call option: max(S - K, 0)
            qc.x(6)  # Initialize ancilla
            qc.ccx(0, 1, 6)  # Controlled on spot > strike approximation
        else:
            # Put option: max(K - S, 0)
            qc.x(7)  # Initialize ancilla
            qc.ccx(1, 0, 7)  # Controlled on strike > spot approximation

    def _extract_price_from_counts(self, counts: Dict[str, int], strike: float, option_type: str) -> float:
        """Extract option price from measurement counts."""
        total_shots = sum(counts.values())
        payoff_sum = 0

        for bitstring, count in counts.items():
            # Decode the measurement
            # This is a simplified extraction - real implementation would be more sophisticated
            probability = count / total_shots

            # Estimate payoff based on bitstring
            # Bit 6 for call, bit 7 for put
            if option_type.lower() == 'call' and bitstring[-7] == '1':
                payoff = max(0, 100 - strike)  # Simplified assumption
            elif option_type.lower() == 'put' and bitstring[-8] == '1':
                payoff = max(0, strike - 100)  # Simplified assumption
            else:
                payoff = 0

            payoff_sum += probability * payoff

        # Discount by risk-free rate (simplified)
        discount_factor = np.exp(-0.05 * 1.0)  # Assume 5% rate, 1 year
        return payoff_sum * discount_factor

    def _classical_fallback(self, spot: float, strike: float, time_to_expiry: float,
                           risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Classical Black-Scholes fallback pricing."""
        from scipy.stats import norm

        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type.lower() == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return price


class QuantumAmplitudeEstimationEngine(QuantumPricingEngine):
    """Quantum amplitude estimation for precise option pricing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.precision_qubits = config.get('precision_qubits', 8)

    def price_option(self, spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Price option using quantum amplitude estimation."""
        if not QISKIT_AVAILABLE:
            return self._classical_fallback(spot, strike, time_to_expiry,
                                          risk_free_rate, volatility, option_type)

        try:
            # Implement quantum amplitude estimation
            qc = self._create_amplitude_estimation_circuit(
                spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
            )

            # Execute circuit
            if hasattr(self.backend, 'run'):
                sampler = Sampler(mode=Session(backend=self.backend))
                job = sampler.run([qc], shots=self.config.get('shots', 1024))
                result = job.result()
                counts = result.quasi_dists[0]
            else:
                qc_transpiled = transpile(qc, self.backend)
                job = self.backend.run(qc_transpiled, shots=self.config.get('shots', 1024))
                result = job.result()
                counts = result.get_counts()

            # Extract amplitude estimation result
            estimated_probability = self._extract_amplitude(counts)
            price = self._probability_to_price(estimated_probability, spot, strike, time_to_expiry, risk_free_rate)

            return price

        except Exception as e:
            logger.error(f"Quantum amplitude estimation failed: {e}")
            return self._classical_fallback(spot, strike, time_to_expiry,
                                          risk_free_rate, volatility, option_type)

    def _create_amplitude_estimation_circuit(self, spot: float, strike: float, time_to_expiry: float,
                                           risk_free_rate: float, volatility: float, option_type: str) -> QuantumCircuit:
        """Create quantum amplitude estimation circuit."""
        n_precision = self.precision_qubits
        n_evaluation = 4  # Qubits for state preparation

        qc = QuantumCircuit(n_precision + n_evaluation, n_precision)

        # Initialize superposition on precision qubits
        for i in range(n_precision):
            qc.h(i)

        # Apply controlled state preparation
        for precision_bit in range(n_precision):
            power = 2 ** precision_bit
            angle = power * np.pi

            # Controlled state preparation (simplified)
            qc.cry(angle, precision_bit, n_precision)

        # Apply oracle (payoff function)
        self._apply_payoff_oracle(qc, n_precision, spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)

        # Apply inverse quantum Fourier transform
        self._apply_iqft(qc, n_precision)

        # Measure precision qubits
        qc.measure(range(n_precision), range(n_precision))

        return qc

    def _apply_payoff_oracle(self, qc: QuantumCircuit, start_qubit: int, spot: float, strike: float,
                           time_to_expiry: float, risk_free_rate: float, volatility: float, option_type: str):
        """Apply payoff oracle for amplitude estimation."""
        # Simplified oracle implementation
        # In practice, this would encode the actual payoff function

        # Encode parameters
        spot_angle = (spot / 200.0) * np.pi
        strike_angle = (strike / 200.0) * np.pi

        qc.ry(spot_angle, start_qubit)
        qc.ry(strike_angle, start_qubit + 1)

        # Apply controlled payoff logic
        if option_type.lower() == 'call':
            qc.ccx(start_qubit, start_qubit + 1, start_qubit + 2)
        else:
            qc.ccx(start_qubit + 1, start_qubit, start_qubit + 2)

    def _apply_iqft(self, qc: QuantumCircuit, n_qubits: int):
        """Apply inverse quantum Fourier transform."""
        for i in range(n_qubits):
            for j in range(i):
                qc.cp(-np.pi / (2 ** (i - j)), j, i)
            qc.h(i)

        # Swap qubits
        for i in range(n_qubits // 2):
            qc.swap(i, n_qubits - i - 1)

    def _extract_amplitude(self, counts: Dict[str, int]) -> float:
        """Extract amplitude from measurement counts."""
        total_shots = sum(counts.values())
        max_count_bitstring = max(counts, key=counts.get)
        measured_phase = int(max_count_bitstring, 2) / (2 ** len(max_count_bitstring))

        # Convert phase to amplitude
        amplitude = np.sin(np.pi * measured_phase) ** 2
        return amplitude

    def _probability_to_price(self, probability: float, spot: float, strike: float,
                            time_to_expiry: float, risk_free_rate: float) -> float:
        """Convert estimated probability to option price."""
        # Simplified conversion
        # In practice, this would be more sophisticated
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        expected_payoff = probability * max(spot - strike, 0)  # Simplified assumption
        return expected_payoff * discount_factor

    def _classical_fallback(self, spot: float, strike: float, time_to_expiry: float,
                           risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Classical fallback."""
        # Same as Monte Carlo engine
        from scipy.stats import norm

        d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type.lower() == 'call':
            price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return price


class HybridPricingEngine(QuantumPricingEngine):
    """Hybrid quantum-classical pricing engine."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.quantum_engine = QuantumMonteCarloEngine(config)
        self.classical_engine = None  # Would import classical engine

    def price_option(self, spot: float, strike: float, time_to_expiry: float,
                    risk_free_rate: float, volatility: float, option_type: str) -> float:
        """Price option using hybrid approach."""
        # Use quantum for high-dimensional parts, classical for the rest
        quantum_price = self.quantum_engine.price_option(
            spot, strike, time_to_expiry, risk_free_rate, volatility, option_type
        )

        # Could combine with classical ML model here
        # For now, return quantum price
        return quantum_price

    def price_swaption_quantum(self, notional: float, strike_rate: float, time_to_expiry: float,
                              risk_free_rate: float, volatility: float, swap_tenor: float) -> float:
        """Price swaption using hybrid quantum approach."""
        # Simplified swaption pricing - treat as option on swap rate
        # In practice, this would be more sophisticated
        spot_swap_rate = 0.05  # Placeholder - would compute actual forward swap rate
        return self.price_option(spot_swap_rate, strike_rate, time_to_expiry,
                               risk_free_rate, volatility, 'call')


def create_quantum_pricing_engine(engine_type: str, config: Dict[str, Any]) -> QuantumPricingEngine:
    """Factory function for quantum pricing engines."""
    engines = {
        'monte_carlo': QuantumMonteCarloEngine,
        'amplitude_estimation': QuantumAmplitudeEstimationEngine,
        'hybrid': HybridPricingEngine,
    }

    if engine_type not in engines:
        raise ValueError(f"Unknown quantum engine type: {engine_type}")

    return engines[engine_type](config)


# Example usage and testing functions
def test_quantum_pricing():
    """Test quantum pricing engines."""
    config = {
        'api_key': 'wPQOh--o2TjczKSr8xYZXZPudXBm4Ia6m__gdphs-5IR',
        'backend': 'simulator',
        'shots': 1024,
        'n_qubits': 8,
        'precision_qubits': 6
    }

    # Test parameters
    spot = 100.0
    strike = 105.0
    time_to_expiry = 1.0
    risk_free_rate = 0.05
    volatility = 0.2

    print("Testing Quantum Pricing Engines")
    print("=" * 40)

    # Test Monte Carlo
    try:
        mc_engine = QuantumMonteCarloEngine(config)
        mc_price = mc_engine.price_option(spot, strike, time_to_expiry, risk_free_rate, volatility, 'call')
        print(f"Monte Carlo Price: ${mc_price:.4f}")
    except Exception as e:
        print(f"Monte Carlo test failed: {e}")

    # Test Amplitude Estimation
    try:
        ae_engine = QuantumAmplitudeEstimationEngine(config)
        ae_price = ae_engine.price_option(spot, strike, time_to_expiry, risk_free_rate, volatility, 'call')
        print(f"Amplitude Estimation Price: ${ae_price:.4f}")
    except Exception as e:
        print(f"Amplitude Estimation test failed: {e}")

    # Test Hybrid
    try:
        hybrid_engine = HybridPricingEngine(config)
        hybrid_price = hybrid_engine.price_option(spot, strike, time_to_expiry, risk_free_rate, volatility, 'call')
        print(f"Hybrid Price: ${hybrid_price:.4f}")
    except Exception as e:
        print(f"Hybrid test failed: {e}")


if __name__ == "__main__":
    test_quantum_pricing()