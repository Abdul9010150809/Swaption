from setuptools import setup, find_packages
setup(
    name="quantum_pricing", 
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pytest", "qiskit", "numpy", "scipy"]
)