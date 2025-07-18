# Contributing to EXOplanet Analysis Project

Thank you for your interest in contributing to the EXOplanet Analysis Project! This document provides guidelines for contributing to this project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Submission Guidelines](#submission-guidelines)
- [Style Guidelines](#style-guidelines)
- [Testing](#testing)

## Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Git
- Basic knowledge of machine learning and data science
- Familiarity with astronomical data (helpful but not required)

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/EXOplanet.git
   cd EXOplanet
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

1. **Bug Reports**: Found a bug? Report it!
2. **Feature Requests**: Have an idea for improvement?
3. **Code Contributions**: Implement new features or fix bugs
4. **Documentation**: Improve existing docs or add new ones
5. **Data Analysis**: Enhance the analysis or add new insights
6. **Visualization**: Create better plots or interactive dashboards

### Reporting Bugs

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages** or stack traces
- **Screenshots** if applicable

Use this template:
```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS, Windows, Linux]
- Python Version: [e.g., 3.8]
- Jupyter Version: [if applicable]
```

### Suggesting Features

When suggesting features:

- **Check existing issues** to avoid duplicates
- **Provide clear use case** and rationale
- **Consider implementation complexity**
- **Think about broader impact** on the project

## Development Setup

### Project Structure
```
EXOplanet/
├── README.md                    # Main documentation
├── requirements.txt             # Dependencies
├── src/                         # Source code (if applicable)
├── notebooks/                   # Jupyter notebooks
├── data/                        # Data files
├── docs/                        # Documentation
├── tests/                       # Test files
└── scripts/                     # Utility scripts
```

### Jupyter Notebook Guidelines

When working with notebooks:

1. **Clear outputs** before committing:
   ```bash
   jupyter nbconvert --clear-output --inplace notebook.ipynb
   ```

2. **Use meaningful cell documentation**:
   - Add markdown cells to explain analysis steps
   - Comment complex code sections
   - Include visualizations with proper titles and labels

3. **Keep notebooks modular**:
   - Break analysis into logical sections
   - Use functions for reusable code
   - Import utility functions from separate modules

## Submission Guidelines

### Pull Request Process

1. **Ensure your code follows** the style guidelines
2. **Update documentation** if necessary
3. **Add tests** for new functionality
4. **Make sure all tests pass**
5. **Update README.md** if needed
6. **Submit pull request** with clear description

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Style Guidelines

### Python Code Style

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specifics:

- **Line length**: 88 characters (Black formatter standard)
- **Imports**: Group imports (standard, third-party, local)
- **Naming conventions**: 
  - snake_case for functions and variables
  - PascalCase for classes
  - UPPER_CASE for constants

### Documentation Style

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain "why" not "what"
- **Markdown**: Use proper headers and formatting

Example docstring:
```python
def calculate_habitability_score(planet_data):
    """Calculate habitability score for an exoplanet.
    
    Args:
        planet_data (dict): Dictionary containing planetary properties.
            Must include 'mass', 'radius', 'temperature', and 'stellar_flux'.
    
    Returns:
        float: Habitability score between 0 and 1, where 1 is most habitable.
    
    Raises:
        ValueError: If required keys are missing from planet_data.
        TypeError: If planet_data is not a dictionary.
    """
```

### Jupyter Notebook Style

- **Cell organization**: One concept per cell
- **Markdown headers**: Use appropriate levels (##, ###, ####)
- **Code comments**: Explain complex operations
- **Visualization**: Include titles, labels, and legends

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_processing.py

# Run with coverage
pytest --cov=src
```

### Writing Tests

- **Test file naming**: `test_<module_name>.py`
- **Test function naming**: `test_<function_being_tested>`
- **Use fixtures** for common test data
- **Test edge cases** and error conditions

Example test:
```python
import pytest
from src.data_processing import clean_exoplanet_data

def test_clean_exoplanet_data_removes_error_columns():
    """Test that error columns are properly removed."""
    # Test implementation here
    pass

def test_clean_exoplanet_data_handles_missing_values():
    """Test missing value handling."""
    # Test implementation here
    pass
```

## Data Science Specific Guidelines

### Exploratory Data Analysis
- **Document assumptions** and decisions
- **Validate findings** with multiple approaches
- **Consider data limitations** and biases
- **Provide scientific context** for astronomical data

### Machine Learning
- **Set random seeds** for reproducibility
- **Use cross-validation** for model evaluation
- **Document hyperparameter choices**
- **Consider class imbalance** in astronomical datasets

### Visualization
- **Use appropriate plot types** for the data
- **Include error bars** where applicable
- **Consider colorblind-friendly palettes**
- **Provide clear legends and labels**

## Getting Help

If you need help:

1. **Check the documentation** first
2. **Search existing issues** on GitHub
3. **Ask questions** in issue discussions
4. **Reach out** to maintainers

## Recognition

Contributors will be acknowledged in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Academic papers** if applicable

Thank you for contributing to the EXOplanet Analysis Project! 🚀
