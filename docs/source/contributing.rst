===============
Contributing
===============

We welcome contributions to the Climate Diagnostics Toolkit! This guide will help you get started.

🤝 **Ways to Contribute**
===========================

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🐛 Report Bugs
      :text-align: center

      Found an issue? Let us know!
      
      +++
      
      Report bugs on our `GitHub Issues <https://github.com/pranay-chakraborty/climate_diagnostics/issues>`_ page.

   .. grid-item-card:: 💡 Suggest Features
      :text-align: center

      Have an idea for improvement?
      
      +++
      
      Start a discussion on our `GitHub Discussions <https://github.com/pranay-chakraborty/climate_diagnostics/discussions>`_ page.

   .. grid-item-card:: 📝 Improve Documentation
      :text-align: center

      Help make our docs better!
      
      +++
      
      Documentation improvements are always welcome.

   .. grid-item-card:: 🔧 Code Contributions
      :text-align: center

      Add new features or fix bugs
      
      +++
      
      Submit pull requests with new functionality.

🚀 **Getting Started**
=======================

**1. Fork and Clone**

.. code-block:: bash

   # Fork on GitHub, then clone your fork
   git clone https://github.com/pranay-chakraborty/climate_diagnostics.git
   cd climate_diagnostics

**2. Set Up Development Environment**

.. code-block:: bash

   # Create conda environment
   conda env create -f environment.yml
   conda activate climate-diagnostics
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install pytest black flake8 sphinx

**3. Create a Branch**

.. code-block:: bash

   git checkout -b feature/your-feature-name

🧪 **Development Workflow**
============================

**Code Style**

We use Black for code formatting:

.. code-block:: bash

   # Format your code
   black src/climate_diagnostics/
   
   # Check formatting
   black --check src/climate_diagnostics/

**Testing**

Run tests before submitting:

.. code-block:: bash

   # Run all tests
   pytest tests/
   
   # Run specific test file
   pytest tests/test_plots.py
   
   # Run with coverage
   pytest --cov=climate_diagnostics tests/

**Documentation**

Build documentation locally:

.. code-block:: bash

   cd docs/
   make html
   
   # View in browser
   open build/html/index.html

📝 **Code Guidelines**
=======================

**Python Style**

- Follow PEP 8
- Use Black for formatting
- Add type hints where appropriate
- Write descriptive docstrings

**Example Function:**

.. code-block:: python

   def calculate_trend(
       data: xr.DataArray,
       dim: str = "time",
       method: str = "linear"
   ) -> xr.DataArray:
       """
       Calculate linear trend along specified dimension.
       
       Parameters
       ----------
       data : xr.DataArray
           Input data array
       dim : str, default "time"
           Dimension along which to calculate trend
       method : str, default "linear"
           Trend calculation method
           
       Returns
       -------
       xr.DataArray
           Trend values with same coordinates as input
           
       Examples
       --------
       >>> trend = calculate_trend(temperature_data)
       """
       # Implementation here
       pass

**Testing Guidelines**

- Write tests for all new functions
- Include edge cases and error conditions
- Use meaningful test names
- Keep tests simple and focused

**Example Test:**

.. code-block:: python

   import pytest
   import numpy as np
   import xarray as xr
   from climate_diagnostics.trends import calculate_trend

   def test_calculate_trend_linear():
       """Test linear trend calculation with known data."""
       # Create test data with known trend
       time = pd.date_range("2000", "2010", freq="YS")
       data = xr.DataArray(
           np.arange(len(time)) + np.random.randn(len(time)) * 0.1,
           dims=["time"],
           coords={"time": time}
       )
       
       trend = calculate_trend(data)
       
       # Should be close to 1.0 per year
       assert abs(trend.values - 1.0) < 0.2

📚 **Documentation Standards**
===============================

**Docstring Format**

Use NumPy-style docstrings:

.. code-block:: python

   def function_name(param1, param2):
       """
       Brief description of the function.
       
       Longer description if needed. Explain what the function
       does, any important algorithms, or usage notes.
       
       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type, optional
           Description of param2 (default: value)
           
       Returns
       -------
       return_type
           Description of return value
           
       Raises
       ------
       ExceptionType
           When this exception is raised
           
       Examples
       --------
       >>> result = function_name(value1, value2)
       >>> print(result)
       Expected output
       """

**Adding Examples**

Include examples in the `examples/` directory:

.. code-block:: python

   """
   Example: Creating Temperature Maps
   =================================
   
   This example shows how to create temperature
   maps using the climate_plots accessor.
   """
   
   import xarray as xr
   import matplotlib.pyplot as plt
   import climate_diagnostics
   
   # Load data
   ds = xr.open_dataset("temperature.nc")
   
   # Create plot
   fig = ds.climate_plots.plot_mean(
       variable="temperature",
       title="Global Mean Temperature"
   )
   plt.show()

🔄 **Pull Request Process**
============================

**1. Prepare Your PR**

.. code-block:: bash

   # Make sure tests pass
   pytest tests/
   
   # Format code
   black src/climate_diagnostics/
   
   # Update documentation if needed
   cd docs && make html

**2. Submit PR**

- Use a descriptive title
- Reference any related issues
- Describe what your changes do
- Include tests for new functionality

**3. PR Template**

.. code-block:: markdown

   ## Description
   Brief description of changes
   
   ## Related Issues
   Fixes #123
   
   ## Changes Made
   - Added new feature X
   - Fixed bug in Y
   - Updated documentation for Z
   
   ## Testing
   - [ ] All tests pass
   - [ ] New tests added for new functionality
   - [ ] Documentation updated
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated

🐛 **Bug Reports**
===================

**Good Bug Report Template:**

.. code-block:: markdown

   ## Bug Description
   Clear description of what the bug is
   
   ## To Reproduce
   1. Load data with `ds = xr.open_dataset(...)`
   2. Call `ds.climate_plots.plot_mean(...)`
   3. See error
   
   ## Expected Behavior
   What you expected to happen
   
   ## Environment
   - OS: [e.g. macOS 12.0]
   - Python version: [e.g. 3.11.0]
   - climate_diagnostics version: [e.g. 1.1.0]
   - xarray version: [e.g. 2023.1.0]
   
   ## Additional Context
   Any other relevant information

💬 **Community Guidelines**
============================

- Be respectful and inclusive
- Help others learn and grow
- Ask questions if you're unsure
- Share your knowledge and experience
- Follow our Code of Conduct

📧 **Getting Help**
====================

- **Questions**: Use GitHub Discussions
- **Bugs**: https://github.com/pranay-chakraborty/climate_diagnostics/issues
- **Features**: Start a discussion first
- **Chat**: Join our community channels

Thank you for contributing to the Climate Diagnostics Toolkit! 🙏
