==========================
Models API Reference
==========================

The models module provides specialized functions for climate model analysis and radiative-convective equilibrium calculations.

Overview
========

The models module contains:

- **RCE (Radiative-Convective Equilibrium)**: Functions for calculating RCE states
- **RE (Radiative Equilibrium)**: Functions for calculating RE states

RCE Module
==========

.. automodule:: climate_diagnostics.models.rce
   :members:
   :undoc-members:

Available Functions
===================

.. autofunction:: climate_diagnostics.models.rce.create_rce_model
   :no-index:

.. autofunction:: climate_diagnostics.models.rce.create_re_model
   :no-index:

Basic Example
=============

.. code-block:: python

   from climate_diagnostics.models.rce import create_rce_model, create_re_model
   
   # Create and run RCE model
   rce_model = create_rce_model(
       num_lev=40,
       water_depth=5.0,
       integrate_years=3
   )

   # Create and run RE model
   re_model = create_re_model(
       num_lev=40,
       water_depth=5.0,
       integrate_years=3
   )

See Also
========

* :doc:`./timeseries` - Time series analysis methods
* :doc:`./trends` - Trend analysis methods
