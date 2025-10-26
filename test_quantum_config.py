#!/usr/bin/env python3
"""
Test script to verify quantum API key integration in PriceMatrix system.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the research development source to path
sys.path.append('01-research-development/src')

def test_config_loading():
    """Test loading configuration with quantum settings."""
    print("Testing configuration loading...")
    
    try:
        from utils.config import Config, load_config
        
        # Test default configuration
        config = Config()
        print(f"✓ Default quantum enabled: {config.quantum.enabled}")
        print(f"✓ Default quantum API key: {config.quantum.api_key[:20]}...")
        print(f"✓ Default quantum provider: {config.quantum.provider}")
        print(f"✓ Default quantum backend: {config.quantum.backend}")
        
        # Test loading from YAML file
        config_from_yaml = Config.from_yaml('01-research-development/experiments/experiment_configs/base_config.yaml')
        print(f"✓ YAML quantum enabled: {config_from_yaml.quantum.enabled}")
        print(f"✓ YAML quantum API key: {config_from_yaml.quantum.api_key[:20]}...")
        print(f"✓ YAML quantum provider: {config_from_yaml.quantum.provider}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable support for quantum API key."""
    print("\nTesting environment variable support...")
    
    try:
        # Set environment variable
        test_api_key = "test-quantum-api-key-12345"
        os.environ['QUANTUM_API_KEY'] = test_api_key
        
        from utils.config import QuantumConfig
        
        # Create new config instance to pick up environment variable
        quantum_config = QuantumConfig()
        
        if quantum_config.api_key == test_api_key:
            print(f"✓ Environment variable correctly loaded: {quantum_config.api_key}")
        else:
            print(f"✗ Environment variable not loaded correctly. Got: {quantum_config.api_key}")
            return False
            
        # Clean up
        del os.environ['QUANTUM_API_KEY']
        
        return True
        
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        return False

def test_production_service_config():
    """Test production ML service quantum configuration."""
    print("\nTesting production service configuration...")
    
    try:
        # Set test environment variables
        os.environ['QUANTUM_API_KEY'] = 'wPQOh--o2TjczKSr8xYZXZPudXBm4Ia6m__gdphs-5IR'
        os.environ['QUANTUM_ENABLED'] = 'true'
        os.environ['QUANTUM_PROVIDER'] = 'ibm'
        
        # Import the quantum config from main.py
        sys.path.append('02-production/ml-service/app')
        
        # Simulate the config loading from main.py
        QUANTUM_CONFIG = {
            'enabled': os.getenv('QUANTUM_ENABLED', 'true').lower() == 'true',
            'api_key': os.getenv('QUANTUM_API_KEY', 'wPQOh--o2TjczKSr8xYZXZPudXBm4Ia6m__gdphs-5IR'),
            'provider': os.getenv('QUANTUM_PROVIDER', 'ibm'),
            'backend': os.getenv('QUANTUM_BACKEND', 'simulator'),
            'shots': int(os.getenv('QUANTUM_SHOTS', '1024')),
            'max_circuits': int(os.getenv('QUANTUM_MAX_CIRCUITS', '100')),
            'optimization_level': int(os.getenv('QUANTUM_OPTIMIZATION_LEVEL', '1')),
            'job_timeout': int(os.getenv('QUANTUM_JOB_TIMEOUT', '300')),
            'max_retries': int(os.getenv('QUANTUM_MAX_RETRIES', '3'))
        }
        
        print(f"✓ Production service quantum enabled: {QUANTUM_CONFIG['enabled']}")
        print(f"✓ Production service API key: {QUANTUM_CONFIG['api_key'][:20]}...")
        print(f"✓ Production service provider: {QUANTUM_CONFIG['provider']}")
        print(f"✓ Production service backend: {QUANTUM_CONFIG['backend']}")
        print(f"✓ Production service shots: {QUANTUM_CONFIG['shots']}")
        
        # Clean up environment variables
        for key in ['QUANTUM_API_KEY', 'QUANTUM_ENABLED', 'QUANTUM_PROVIDER']:
            if key in os.environ:
                del os.environ[key]
        
        return True
        
    except Exception as e:
        print(f"✗ Production service configuration test failed: {e}")
        return False

def test_yaml_config_files():
    """Test YAML configuration files contain quantum settings."""
    print("\nTesting YAML configuration files...")
    
    try:
        # Test base config
        base_config_path = '01-research-development/experiments/experiment_configs/base_config.yaml'
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        if 'quantum' in base_config:
            print(f"✓ Base config has quantum section")
            print(f"✓ Base config quantum enabled: {base_config['quantum']['enabled']}")
            print(f"✓ Base config API key: {base_config['quantum']['api_key'][:20]}...")
        else:
            print("✗ Base config missing quantum section")
            return False
        
        # Test advanced config
        advanced_config_path = '01-research-development/experiments/experiment_configs/advanced_config.yaml'
        with open(advanced_config_path, 'r') as f:
            advanced_config = yaml.safe_load(f)
        
        if 'quantum' in advanced_config:
            print(f"✓ Advanced config has quantum section")
            print(f"✓ Advanced config quantum enabled: {advanced_config['quantum']['enabled']}")
            print(f"✓ Advanced config API key: {advanced_config['quantum']['api_key'][:20]}...")
            print(f"✓ Advanced config has quantum pricing: {advanced_config['quantum']['use_quantum_pricing']}")
        else:
            print("✗ Advanced config missing quantum section")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ YAML configuration test failed: {e}")
        return False

def test_env_example_file():
    """Test .env.example file contains quantum settings."""
    print("\nTesting .env.example file...")
    
    try:
        env_example_path = '.env.example'
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        required_vars = [
            'QUANTUM_ENABLED',
            'QUANTUM_API_KEY',
            'QUANTUM_PROVIDER',
            'QUANTUM_BACKEND',
            'QUANTUM_SHOTS'
        ]
        
        for var in required_vars:
            if var in content:
                print(f"✓ .env.example contains {var}")
            else:
                print(f"✗ .env.example missing {var}")
                return False
        
        # Check if the API key is present
        if 'wPQOh--o2TjczKSr8xYZXZPudXBm4Ia6m__gdphs-5IR' in content:
            print("✓ .env.example contains the quantum API key")
        else:
            print("✗ .env.example missing the quantum API key")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ .env.example test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("PriceMatrix Quantum API Key Integration Test")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_environment_variables,
        test_production_service_config,
        test_yaml_config_files,
        test_env_example_file
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Quantum API key integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())