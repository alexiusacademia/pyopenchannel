# Contributing to PyOpenChannel

Thank you for your interest in contributing to PyOpenChannel! This document provides guidelines and information for contributors.

## Getting Started

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/alexiusacademia/pyopenchannel.git
   cd pyopenchannel
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests**
   ```bash
   pytest
   ```

4. **Run Code Quality Checks**
   ```bash
   black src/ tests/ examples/
   isort src/ tests/ examples/
   flake8 src/ tests/ examples/
   mypy src/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

### Python Code Style

- **PEP 8**: Follow Python PEP 8 style guidelines
- **Line Length**: Maximum 88 characters (Black default)
- **Imports**: Use isort for import organization
- **Type Hints**: Include type hints for all public functions
- **Docstrings**: Use Google-style docstrings

### Example Function

```python
def calculate_normal_depth(
    channel: ChannelGeometry,
    discharge: float,
    slope: float,
    manning_n: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = MAX_ITERATIONS
) -> float:
    """
    Calculate normal depth for uniform flow using Newton-Raphson method.
    
    Args:
        channel: Channel geometry object
        discharge: Discharge in m³/s
        slope: Channel slope (dimensionless)
        manning_n: Manning's roughness coefficient
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        Normal depth in meters
        
    Raises:
        ConvergenceError: If solution doesn't converge
        InvalidFlowConditionError: If flow conditions are invalid
        
    Example:
        >>> channel = RectangularChannel(width=3.0)
        >>> depth = calculate_normal_depth(channel, 5.0, 0.001, 0.025)
        >>> print(f"Normal depth: {depth:.3f} m")
    """
    # Implementation here
    pass
```

### Documentation Style

- **Docstrings**: Complete docstrings for all public classes and functions
- **Type Information**: Include parameter and return types
- **Examples**: Provide usage examples where helpful
- **Error Conditions**: Document exceptions that may be raised

## Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Property Tests**: Test mathematical properties and relationships
- **Edge Cases**: Test boundary conditions and error cases

### Test Organization

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and configuration
├── test_geometry.py         # Channel geometry tests
├── test_hydraulics.py       # Hydraulic calculation tests
├── test_flow_analysis.py    # Flow analysis tests
├── test_design.py           # Design module tests
├── test_validators.py       # Input validation tests
└── test_integration.py      # Integration tests
```

### Writing Tests

```python
def test_rectangular_channel_area():
    """Test rectangular channel area calculation."""
    channel = RectangularChannel(width=4.0)
    
    # Test normal case
    assert channel.area(depth=2.0) == 8.0
    
    # Test edge case
    assert channel.area(depth=0.1) == 0.4
    
    # Test error condition
    with pytest.raises(InvalidFlowConditionError):
        channel.area(depth=-1.0)
```

### Test Coverage

- Aim for >90% test coverage
- Test both success and failure cases
- Include edge cases and boundary conditions
- Test numerical accuracy for hydraulic calculations

## Contributing Areas

### High Priority

1. **Bug Fixes**: Fix any identified issues
2. **Documentation**: Improve documentation and examples
3. **Test Coverage**: Add tests for uncovered code
4. **Performance**: Optimize critical calculations

### Medium Priority

1. **New Channel Shapes**: Add support for additional geometries
2. **Advanced Calculations**: Implement gradually varied flow
3. **Validation**: Add more comprehensive input validation
4. **Examples**: Create more real-world examples

### Future Enhancements

1. **Visualization**: Add plotting capabilities
2. **Optimization**: Advanced optimization algorithms
3. **File I/O**: Support for data import/export
4. **GUI**: Simple graphical interface

## Hydraulic Engineering Guidelines

### Physical Accuracy

- **Units**: Use consistent SI units (meters, m³/s, etc.)
- **Validation**: Validate against known solutions
- **Convergence**: Ensure numerical methods converge reliably
- **Physical Limits**: Check for physically reasonable results

### Mathematical Implementation

- **Numerical Methods**: Use robust numerical algorithms
- **Convergence Criteria**: Appropriate tolerance values
- **Initial Guesses**: Good starting values for iterations
- **Error Handling**: Graceful handling of edge cases

### Engineering Practices

- **Safety Factors**: Consider appropriate safety margins
- **Design Standards**: Follow established design practices
- **Practical Limits**: Implement reasonable parameter ranges
- **User Guidance**: Provide helpful error messages

## Documentation Contributions

### Types of Documentation

1. **API Documentation**: Function and class documentation
2. **User Guide**: How-to guides and tutorials
3. **Examples**: Practical usage examples
4. **Theory**: Background and theoretical explanations

### Documentation Standards

- **Clarity**: Write for the target audience
- **Completeness**: Cover all important aspects
- **Accuracy**: Ensure technical accuracy
- **Examples**: Include practical examples

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Changes are tested
- [ ] Commit messages are clear

### Pull Request Description

Include:
- **Summary**: Brief description of changes
- **Motivation**: Why the change is needed
- **Testing**: How the change was tested
- **Breaking Changes**: Any breaking changes
- **Related Issues**: Link to related issues

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainer reviews code
3. **Discussion**: Address any feedback
4. **Approval**: Maintainer approves changes
5. **Merge**: Changes are merged to main branch

## Issue Reporting

### Bug Reports

Include:
- **Description**: Clear description of the bug
- **Reproduction**: Steps to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, etc.
- **Code Sample**: Minimal example demonstrating the bug

### Feature Requests

Include:
- **Description**: Clear description of the feature
- **Use Case**: Why the feature is needed
- **Implementation**: Suggested implementation approach
- **Examples**: Usage examples

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

### Getting Help

- **Documentation**: Check existing documentation first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions noted in releases
- **Documentation**: Contributors credited in docs

## License

By contributing to PyOpenChannel, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to PyOpenChannel! Your contributions help make hydraulic engineering more accessible and efficient.
