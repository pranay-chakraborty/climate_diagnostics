import numpy as np
import climlab
from climlab.radiation.radiation import default_absorbers
from climlab import constants as const
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # For advanced signal smoothing to reduce numerical noise


def create_rce_model(num_lev=60,
                     water_depth=5.,
                     integrate_years=5,
                     gas_vmr=None,
                     savgol_window=15,
                     savgol_order=3):
   """ Simulates single-column Radiative-Convective Equilibrium (RCE).

   This function sets up and runs a single-column atmospheric model until it
   reaches RCE. It uses the CAM3 radiation scheme and a simple
   Convective Adjustment scheme, which is climlab's default.

   The final temperature profile is smoothed using a Savitzky-Golay filter
   and plotted.

   Args:
       num_lev (int): The number of vertical levels in the atmosphere.
       water_depth (float): The depth of the slab ocean surface model (in meters),
                            which determines the heat capacity of the surface.
       integrate_years (float): The number of years to run the simulation to approach
                                equilibrium.
       gas_vmr (dict, optional): A dictionary to override default greenhouse gas
                                 volume mixing ratios (e.g., {'CO2': 400e-6}).
                                 If None, climlab defaults are used.
       savgol_window (int): The window size for the Savitzky-Golay filter.
                            Must be an odd integer.
       savgol_order (int): The polynomial order for the Savitzky-Golay filter.
                           Must be less than savgol_window.

   Returns:
       climlab.TimeDependentProcess: The model object after the integration is
                                     complete, containing all state variables
                                     and diagnostics.
   """
   
   # ========================================================================
   # STEP 1: Initialize the atmospheric state
   # ========================================================================
   
   # Create a vertical column with specified number of levels and slab ocean
   # The slab ocean provides thermal inertia at the surface
   state = climlab.column_state(num_lev=num_lev, water_depth=water_depth)

   # ========================================================================
   # STEP 2: Configure greenhouse gas concentrations
   # ========================================================================
   
   # Get default concentrations for all greenhouse gases (CO2, CH4, N2O, etc.)
   absorber_vmr = default_absorbers(state.Tatm)
   
   # Override default gas concentrations with custom values if provided
   if gas_vmr:
       absorber_vmr.update(gas_vmr)

   # ========================================================================
   # STEP 3: Set up water vapor feedback process
   # ========================================================================
   
   # Create interactive water vapor that responds to temperature changes
   # This is essential for realistic climate sensitivity
   h2o = climlab.radiation.ManabeWaterVapor(state=state, name='H2O')

   # ========================================================================
   # STEP 4: Configure radiation scheme
   # ========================================================================
   
   # Set up CAM3 radiation scheme with specified greenhouse gases
   # Links to the dynamic water vapor from step 3
   rad = climlab.radiation.CAM3(
       state=state,
       name='Radiation',
       specific_humidity=h2o.q,  # Use dynamic humidity from water vapor process
       absorber_vmr=absorber_vmr,  # Use greenhouse gas concentrations from step 2
       timestep=const.seconds_per_day  # Calculate radiation once per day
   )

   # ========================================================================
   # STEP 5: Set up convection scheme
   # ========================================================================
   
   # Configure convective adjustment to maintain atmospheric stability
   # Prevents unrealistic super-adiabatic lapse rates
   conv = climlab.convection.ConvectiveAdjustment(
       state=state,
       adj_lapse_rate=6.5,  # Target lapse rate: 6.5 K/km (typical tropospheric value)
       timestep=const.seconds_per_hour  # Check for instability every hour
   )

   # ========================================================================
   # STEP 6: Assemble the complete model
   # ========================================================================
   
   # Create main model container with hourly timestep
   model = climlab.TimeDependentProcess(state=state,
                                        timestep=const.seconds_per_hour)
   
   # Add all physics components as coupled subprocesses
   model.add_subprocess('Radiation', rad)      # Step 4: Radiative heating/cooling
   model.add_subprocess('Convection', conv)    # Step 5: Convective mixing
   model.add_subprocess('WaterVapor', h2o)     # Step 3: Humidity adjustment

   # ========================================================================
   # STEP 7: Run the model to equilibrium
   # ========================================================================
   
   print("Starting model integration with default convection...")
   # Integrate forward in time until radiative-convective equilibrium is reached
   model.integrate_years(integrate_years)
   print("Integration complete. Plotting results.")

   # ========================================================================
   # STEP 8: Process and smooth the temperature profile
   # ========================================================================
   
   # Extract final atmospheric temperature profile
   T_raw = model.Tatm
   
   # Apply Savitzky-Golay filter to smooth numerical noise while preserving features
   # This polynomial filter is superior to simple moving averages for scientific data
   T_smooth = savgol_filter(T_raw, window_length=savgol_window, polyorder=savgol_order)

   # ========================================================================
   # STEP 9: Create publication-quality plot
   # ========================================================================
   
   # Set up high-resolution figure
   fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
   
   # Plot raw data points to show model resolution
   ax.semilogy(T_raw, model.lev, 'o', ms=4,
               label='Raw Levels', alpha=0.5)
   
   # Plot smoothed profile as primary result
   ax.semilogy(T_smooth, model.lev, '-', lw=2, color='crimson',
               label=f'Savitzky-Golay (w={savgol_window}, p={savgol_order})')
   
   # Configure axes: pressure decreases upward (atmospheric convention)
   ax.invert_yaxis()
   ax.set_xlabel('Temperature (K)', fontsize=14)
   ax.set_ylabel('Pressure (hPa)', fontsize=14)
   ax.set_title('RCE Temperature Profile', fontsize=16)
   ax.grid(True)
   ax.legend(loc='upper right', fontsize=12)
   plt.tight_layout()
   plt.show()

   # ========================================================================
   # STEP 10: Return the complete model for further analysis
   # ========================================================================
   
   return model
