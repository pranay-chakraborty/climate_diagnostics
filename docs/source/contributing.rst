Contributing
===========

Thank you for your interest in contributing to the Climate Diagnostics Toolkit! This document provides guidelines and instructions for contributing to the project.

Getting Started
--------------

Prerequisites
~~~~~~~~~~~~

Before you begin, ensure you have the following installed:

* Python 3.10 or higher
* Git
* A virtual environment tool (venv, conda, etc.)

Setting Up Your Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/library.git
      cd library

3. Set up a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install the package in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

Development Workflow
-------------------

Coding Standards
~~~~~~~~~~~~~~~

* Follow PEP 8 style guidelines
* Use descriptive variable names
* Include docstrings for all functions, classes, and modules using NumPy docstring format
* Add type hints where appropriate

Making Changes
~~~~~~~~~~~~~

1. Create a new branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

2. Make your changes
3. Add tests for your changes
4. Update documentation as needed
5. Ensure all tests pass

Testing
~~~~~~~

Run the test suite with:

.. code-block:: bash

   pytest tests/

For coverage reports:

.. code-block:: bash

   pytest --cov=climate_diagnostics tests/

Building Documentation
~~~~~~~~~~~~~~~~~~~~

The documentation is built using Sphinx:

.. code-block:: bash

   cd docs
   make html

You can view the documentation by opening ``docs/build/html/index.html`` in your browser.

Submitting Changes
-----------------

1. Update your branch with the latest changes from main:

   .. code-block:: bash

      git fetch origin
      git rebase origin/main

2. Push your changes to your fork:

   .. code-block:: bash

      git push origin feature-name

3. Submit a pull request through GitHub
4. Include a clear description of the changes and any relevant issue numbers

Pull Request Guidelines
----------------------

* Include tests for new features
* Update documentation for API changes
* Follow the existing code style
* Keep pull requests focused on a single topic
* Reference any related issues

Documentation
------------

* Update API documentation for new features or changes
* Add examples where appropriate
* Make sure your docstrings include parameters, return values, and examples

Questions?
---------

If you have questions about contributing, open an issue with the "question" label or contact the maintainers directly.

Thank you for contributing to the Climate Diagnostics Toolkit!