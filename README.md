# Price Matrix

A comprehensive financial pricing system for derivatives, featuring research and development, production deployment, and complete documentation.

## Overview

This project implements a price matrix system for financial instruments, particularly focusing on swaption pricing using machine learning models. The system is structured in three main phases:

1. **Research & Development** - Model experimentation and validation
2. **Production** - Deployed services and infrastructure
3. **Documentation** - API references and user guides

## Architecture

### Research & Development (01-research-development/)
- Data generation and preprocessing
- Feature engineering
- Model training (Random Forest, Neural Networks, Ensemble)
- Pricing algorithms (Analytic, Monte Carlo)
- Risk metrics calculation
- Experiment tracking and evaluation

### Production (02-production/)
- **API Gateway** - Node.js/Express service for request routing
- **Frontend** - React application for user interface
- **ML Service** - FastAPI service for model inference
- **Infrastructure** - Docker, Kubernetes, monitoring setup

### Documentation (03-documentation/)
- API specifications (Swagger, Postman)
- Model documentation and risk assessment
- User guides and troubleshooting

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker
- Kubernetes (for production deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pricematrix
```

2. Set up research environment:
```bash
cd 01-research-development
pip install -r requirements.txt
```

3. Set up production services:
```bash
cd ../02-production
# Follow individual service READMEs
```

## Usage

### Research
Run notebooks in `01-research-development/notebooks/` for data exploration, model training, and evaluation.

### Production
Use Docker Compose for local deployment:
```bash
cd 02-production/infrastructure
docker-compose up
```

Access the frontend at `http://localhost:3000`

## API Documentation

See `03-documentation/api/` for complete API specifications.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.