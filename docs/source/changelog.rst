===============
Changelog
===============

All notable changes to the Climate Diagnostics Toolkit will be documented here.

🚀 **Version 1.1.1** (2025-07-02)
=================================

**Major Features**

- 🚀 **Sophisticated Disk-Aware Chunking Strategy** - Advanced chunking utilities with memory optimization and disk-aware processing
- ⚡ **Dynamic Chunk Calculator** - Automatically optimizes chunks based on operation type and system resources  
- 🔧 **Performance Profiling** - Built-in chunking analysis and optimization recommendations
- 🧠 **Memory-Conscious Processing** - Intelligent memory estimation and chunk size calculation
- 📊 **Operation-Specific Optimization** - Tailored chunking for time series, spatial analysis, and trend calculations

**Enhanced Features**

- ✨ Complete documentation overhaul with beautiful Furo theme
- 🗺️ Enhanced plotting capabilities with Cartopy integration
- 📈 Advanced time series decomposition methods with optimized chunking
- 📊 Statistical significance testing for trends with spatial chunking
- ⚡ Advanced Dask integration for large dataset processing
- 🎨 Customizable plot styling options

🔄 **Version 1.1.0** (2025-06-30)
=================================

**Legacy Features**

- ✨ Basic documentation and plotting capabilities
- 🗺️ Initial Cartopy integration
- 📈 Basic time series decomposition methods
- 📊 Initial statistical significance testing for trends
- ⚡ Basic Dask integration for large dataset processing
- 🎨 Basic plot styling options

**Improvements**

- 🔧 Better error handling and user feedback
- 📚 Comprehensive API documentation
- 🎯 Improved performance for spatial calculations
- 🌍 Support for multiple coordinate systems
- 📱 Responsive documentation design

**Bug Fixes**

- 🐛 Fixed coordinate handling for irregular grids
- 🔧 Resolved memory issues with large datasets
- 📊 Corrected trend calculation edge cases
- 🗺️ Fixed projection issues in polar regions

**Documentation**

- 📖 New tutorial series for beginners
- 🎓 Advanced user guides and examples
- 🔗 Interactive code examples
- 📝 Contributing guidelines and development setup

🔄 **Version 1.0.0** (2025-01-01)
=================================

**Initial Release**

- 🎉 First public release of Climate Diagnostics Toolkit
- 🗺️ Basic plotting functionality
- 📈 Time series analysis tools
- 📊 Trend calculation methods
- 🔧 Utility functions for climate data

**Core Features**

- xarray accessor integration
- Geographic visualization support
- Statistical analysis tools
- Climate index calculations

📋 **Release Notes**
=====================

**Upcoming Features (v1.2.0)**

- 🤖 Machine learning integration for pattern detection
- 🌐 Web-based interactive plotting
- 📊 Enhanced statistical diagnostics
- 🔄 Improved data format support
- ⚡ Performance optimizations

**Long-term Roadmap**

- 🎯 Real-time data processing capabilities
- 🌍 Climate model evaluation tools
- 📱 Mobile-friendly documentation
- 🤝 Community plugin system

📅 **Release Schedule**
========================

We follow semantic versioning (SemVer) and aim for:

- **Major releases**: Annually (breaking changes)
- **Minor releases**: Quarterly (new features)
- **Patch releases**: As needed (bug fixes)

🏷️ **Version Numbering**
========================

Our version numbers follow the format: ``MAJOR.MINOR.PATCH``

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

📊 **Migration Guides**
========================

**Upgrading from v1.0.x to v1.1.x**

No breaking changes! All v1.0 code should work without modification.

**New Features Available:**

.. code-block:: python

   # New in v1.1: Enhanced plotting options
   fig = ds.climate_plots.plot_mean(
       variable="temperature",
       projection="Robinson",  # New projections
       significance_test=True,  # New feature
       colorbar_extend="both"   # Enhanced styling
   )

**Deprecated Features**

- ``old_plot_function()`` → Use ``plot_mean()`` instead
- ``legacy_trend_calc()`` → Use ``calculate_spatial_trends()`` instead

🔗 **Links**
=============

- `GitHub Releases <https://github.com/pranay-chakraborty/climate_diagnostics/releases>`_
- **GitHub Repository**: https://github.com/pranay-chakraborty/climate_diagnostics
- **Documentation**: https://pranay-chakraborty.github.io/climate_diagnostics/
