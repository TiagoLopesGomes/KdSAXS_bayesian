"""
KdSAXS_bayes: Bayesian Analysis of SAXS Data for Protein Oligomerization

This module implements Bayesian analysis of Small Angle X-ray Scattering (SAXS) data
to determine dissociation constants (Kd) for protein oligomerization states.

The analysis workflow includes:
1. Loading and processing SAXS data
2. Solving monomer-oligomer equilibrium equations
3. Calculating chi-squared values using ATSAS oligomer
4. Performing Bayesian MCMC analysis
5. Generating publication-quality plots

Classes
-------
EquilibriumSolver
    Solves chemical equilibrium equations for monomer-oligomer systems
SAXSDataHandler
    Handles loading and processing of SAXS data
BayesianSAXSAnalysis
    Performs Bayesian analysis of SAXS data
SAXSPlotter
    Generates publication-quality plots of analysis results

Notes
-----
Requires ATSAS software suite for SAXS data processing.
Uses PyMC for Bayesian analysis and Plotly for visualization.

Author: [Your Name]
Date: [Current Date]
Version: 1.0.0
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from scipy.optimize import fsolve
import subprocess
import re
import warnings
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

################################################# CONFIGURATION #################################

# Paths
ATSAS_PATH = Path("/Users/tiago/ATSAS-3.2.1-1/bin/")
BASE_DIR = Path(__file__).parent  # Gets the directory where the script is located
THEO_DIR = BASE_DIR / 'examples' / 'blg' / 'ph7' / 'theoretical_saxs'
EXP_DIR = BASE_DIR / 'examples' / 'blg' / 'ph7' / 'exp_saxs'

# Theoretical SAXS data files
THEO_MON_NAME = 'avg_mon_ph7.int'
THEO_DIM_NAME = 'avg_dim_ph7.int'
THEO_MON_FILE = THEO_DIR / THEO_MON_NAME
THEO_DIM_FILE = THEO_DIR / THEO_DIM_NAME

# Experimental conditions
CONCENTRATIONS = [17.4, 26.1, 34.8, 52.2, 69.6, 78.3, 104.3, 130.4, 156.5, 260.9]
EXP_FILES = ['0.32_mgml_cut_28.dat', '0.48_mgml_cut_28.dat', '0.64_mgml_cut_28.dat', 
             '0.96_mgml_cut_28.dat', '1.28_mgml_cut_28.dat', '1.44_mgml_cut_28.dat',
             '1.92_mgml_cut_28.dat', '2.4_mgml_cut_28.dat', '2.88_mgml_cut_28.dat',
             '4.8_mgml_cut_28.dat']

# Analysis parameters
KD_RANGE = (0.1, 10000)  # µM
KD_POINTS = 40
N_VALUE = 2  # Stoichiometry
CHI2_THRESHOLD = 2

# Output directories
OUTPUT_DIRS = {
    'theoretical': 'theoretical_int',
    'fits': 'fits',
    'logs': 'logs',
    'figures': 'figures'
}

# Concentration range for molecular fractions
CONC_RANGE = (0.1, 12000)  # µM
CONC_POINTS = 50

# MCMC parameters
MCMC_PARAMS = {
    'samples': 10000,
    'tune': 5000,
    'chains': 10,
    'target_accept': 0.95
}

################################################# BASE CLASSES #################################

class EquilibriumSolver:
    """
    Solves chemical equilibrium equations for monomer-oligomer systems.

    This class implements methods to solve the equilibrium equations for
    protein oligomerization and calculate molecular fractions.

    Parameters
    ----------
    n_value : int
        Stoichiometry of oligomerization (e.g., 2 for dimer)

    Attributes
    ----------
    n_value : int
        Stored oligomerization stoichiometry

    Notes
    -----
    Uses the following equilibrium equations:
    O * Kd = M^n  (equilibrium)
    M + n*O = Ctot  (mass conservation)
    where M = monomer, O = oligomer, n = stoichiometry
    """
    
    def __init__(self, n_value: float):
        """
        Initialize the EquilibriumSolver.

        Parameters
        ----------
        n_value : int
            Stoichiometry of oligomerization (must be >= 2)

        Raises
        ------
        ValueError
            If n_value is less than 2
        """
        
        self.n_value = n_value
    
    def solve_system(self, concentration: float, kd: float) -> Tuple[float, float]:
        """
        Solve the equilibrium equations for a monomer-oligomer equilibrium.

        Parameters
        ----------
        concentration : float
            Total protein concentration in µM
        kd : float
            Dissociation constant in µM

        Returns
        -------
        Tuple[float, float]
            Monomer and oligomer concentrations (M, O)
            Returns (np.nan, np.nan) if no valid solution is found

        Notes
        -----
        Uses scipy.optimize.fsolve to solve the system of equations:
        O * Kd = M^n
        M + n*O = Ctot

        Examples
        --------
        >>> solver = EquilibriumSolver(n_value=2)
        >>> M, O = solver.solve_system(concentration=100, kd=50)
        """
        def equations(vars):
            M, O = vars
            eq1 = O * kd - M**self.n_value
            eq2 = concentration - (M + self.n_value * O)
            return [eq1, eq2]
        
        initial_guesses = [concentration, concentration / (3 * self.n_value)]
        solution = fsolve(equations, initial_guesses)
        
        return tuple(solution) if all(x >= 0 for x in solution) else (np.nan, np.nan)
    
    def calculate_fractions(self, kd: float, concentration_range: np.ndarray) -> pd.DataFrame:
        """
        Calculate molecular fractions across a range of concentrations.

        Parameters
        ----------
        kd : float
            Dissociation constant in µM
        concentration_range : np.ndarray
            Array of concentrations to evaluate

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - concentration: protein concentration in µM
            - monomer_fraction: fraction of protein in monomer state
            - oligomer_fraction: fraction of protein in oligomer state

        Notes
        -----
        Fractions are calculated using the solve_system method for each
        concentration in the range.
        """
        fractions = []
        for concentration in concentration_range:
            M, O = self.solve_system(concentration, kd)
            if not np.isnan(M):
                monomer_fraction = M / concentration
                oligomer_fraction = self.n_value * O / concentration
                fractions.append((concentration, monomer_fraction, oligomer_fraction))
            else:
                fractions.append((concentration, np.nan, np.nan))
        
        return pd.DataFrame(fractions, 
                          columns=['concentration', 'monomer_fraction', 
                                 'oligomer_fraction'])

class SAXSDataHandler:
    """
    Handles loading and processing of SAXS data.

    This class manages the loading, validation, and processing of both
    theoretical and experimental SAXS data files. It also handles the
    creation of necessary output directories.

    Parameters
    ----------
    theo_mon_file : Path
        Path to theoretical monomer SAXS data file
    theo_dim_file : Path
        Path to theoretical dimer SAXS data file

    Attributes
    ----------
    theo_mon_data : np.ndarray
        Loaded theoretical monomer SAXS data
    theo_dim_data : np.ndarray
        Loaded theoretical dimer SAXS data

    Raises
    ------
    FileNotFoundError
        If theoretical data files are not found
    """
    
    def __init__(self, theo_mon_file: Path, theo_dim_file: Path):
        """
        Initialize the SAXSDataHandler.

        Parameters
        ----------
        theo_mon_file : Path
            Path to theoretical monomer data file
        theo_dim_file : Path
            Path to theoretical dimer data file

        Raises
        ------
        FileNotFoundError
            If either theoretical file is not found
        """
        self._validate_paths(theo_mon_file, theo_dim_file)
        self.theo_mon_data = self._load_theoretical_data(theo_mon_file)
        self.theo_dim_data = self._load_theoretical_data(theo_dim_file)
        
        # Create necessary directories
        for dir_name in OUTPUT_DIRS.values():
            os.makedirs(dir_name, exist_ok=True)
    
    @staticmethod
    def _validate_paths(theo_mon_file: Path, theo_dim_file: Path) -> None:
        """
        Validate existence of theoretical data files.

        Parameters
        ----------
        theo_mon_file : Path
            Path to theoretical monomer data file
        theo_dim_file : Path
            Path to theoretical dimer data file

        Raises
        ------
        FileNotFoundError
            If either file doesn't exist
        """
        if not theo_mon_file.exists():
            raise FileNotFoundError(f"Monomer file not found: {theo_mon_file}")
        if not theo_dim_file.exists():
            raise FileNotFoundError(f"Dimer file not found: {theo_dim_file}")
    
    @staticmethod
    def _load_theoretical_data(filepath: Path) -> np.ndarray:
        """
        Load theoretical SAXS data from file.

        Parameters
        ----------
        filepath : Path
            Path to theoretical data file

        Returns
        -------
        np.ndarray
            Loaded SAXS data with header removed

        Raises
        ------
        Exception
            If there's an error loading the data
        """
        try:
            return np.loadtxt(filepath, skiprows=1)
        except Exception as e:
            logger.error(f"Error loading theoretical data: {e}")
            raise
    
    @staticmethod
    def load_experimental_data(filepath: Path) -> np.ndarray:
        """
        Load experimental SAXS data from file.

        Parameters
        ----------
        filepath : Path
            Path to experimental data file

        Returns
        -------
        np.ndarray
            Processed SAXS data with header and footer removed

        Raises
        ------
        Exception
            If there's an error loading the data

        Notes
        -----
        Assumes data format with 2-line header and 2-line footer
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            return np.loadtxt(lines[2:-2])
        except Exception as e:
            logger.error(f"Error loading experimental data: {e}")
            raise
    
    def calculate_theoretical_intensity(self, monomer_fraction: float, 
                                     oligomer_fraction: float) -> np.ndarray:
        """
        Calculate theoretical SAXS intensity for a mixture of species.

        Parameters
        ----------
        monomer_fraction : float
            Fraction of protein in monomer state (0 to 1)
        oligomer_fraction : float
            Fraction of protein in oligomer state (0 to 1)

        Returns
        -------
        np.ndarray
            Combined theoretical SAXS intensity

        Notes
        -----
        Calculates intensity as a linear combination:
        I(q) = f_mon * I_mon(q) + f_oli * I_oli(q)
        where f_mon and f_oli are the fractions of each species
        """
        return (monomer_fraction * self.theo_mon_data + 
                oligomer_fraction * self.theo_dim_data)

class BayesianSAXSAnalysis:
    """
    Performs Bayesian analysis of SAXS data to determine oligomerization Kd.
    
    This class implements MCMC sampling to determine protein oligomerization
    dissociation constants from SAXS data, including uncertainty estimation
    and model validation.
    
    Parameters
    ----------
    experimental_files : List[Path]
        List of paths to experimental SAXS data files
    concentrations : List[float]
        List of protein concentrations in µM
    data_handler : SAXSDataHandler
        Handler for SAXS data processing
    n_value : int, optional
        Stoichiometry of oligomerization (default: 2)
    kd_range : Tuple[float, float], optional
        Range for Kd exploration (min, max) in µM
    mcmc_params : Dict, optional
        MCMC sampling parameters
        
    Attributes
    ----------
    experimental_files : List[Path]
        Stored paths to experimental files
    concentrations : np.ndarray
        Array of protein concentrations
    data_handler : SAXSDataHandler
        SAXS data handler instance
    n_value : int
        Oligomerization stoichiometry
    kd_range : Tuple[float, float]
        Current Kd range for analysis
    trace : arviz.InferenceData
        MCMC sampling results
    kd_samples : np.ndarray
        Posterior samples of Kd values
    
    Notes
    -----
    Uses PyMC for MCMC sampling with:
    - Log-uniform prior for Kd
    - Custom likelihood based on chi-squared values
    - Multiple chains for convergence assessment
    """
    
    def __init__(self, experimental_files: List[Path], concentrations: List[float],
                 data_handler: SAXSDataHandler, n_value: int = N_VALUE,
                 kd_range: Tuple[float, float] = KD_RANGE,
                 mcmc_params: Dict = MCMC_PARAMS):
        """
        Initialize Bayesian SAXS analysis.

        Parameters
        ----------
        experimental_files : List[Path]
            List of paths to experimental SAXS data files
        concentrations : List[float]
            List of protein concentrations in µM
        data_handler : SAXSDataHandler
            Handler for SAXS data processing
        n_value : int, optional
            Stoichiometry of oligomerization (default: 2)
        kd_range : Tuple[float, float], optional
            Range for Kd exploration (min, max) in µM
        mcmc_params : Dict, optional
            MCMC sampling parameters

        Raises
        ------
        ValueError
            If number of files doesn't match number of concentrations
            If Kd range is invalid
        """
        self.experimental_files = experimental_files
        self.concentrations = np.array(concentrations)
        self.data_handler = data_handler
        self.n_value = n_value
        self.kd_range = kd_range
        self.mcmc_params = mcmc_params
        
        self.equilibrium_solver = EquilibriumSolver(n_value)
        self.trace = None
        self.kd_samples = None
        
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """
        Validate input parameters.

        Raises
        ------
        ValueError
            If number of files doesn't match number of concentrations
            If Kd range is invalid
        """
        if len(self.experimental_files) != len(self.concentrations):
            raise ValueError("Number of files must match number of concentrations")
        if self.kd_range[0] >= self.kd_range[1]:
            raise ValueError("Invalid Kd range: min must be less than max")
    
    def compute_chi_squared_for_kd(self, kd_value: Union[float, pt.TensorVariable]) -> Union[np.ndarray, pt.TensorVariable]:
        """Compute chi-squared values for given Kd across all concentrations."""
        chi_squared_values = []
        
        try:
            kd_val = float(kd_value.eval() if hasattr(kd_value, 'eval') else kd_value)
        except Exception as e:
            logger.error(f"Error converting Kd value: {e}")
            raise
        
        for exp_file, conc in zip(self.experimental_files, self.concentrations):
            try:
                chi_squared = self._compute_single_chi_squared(exp_file, conc, kd_val)
                chi_squared_values.append(chi_squared)
            except Exception as e:
                logger.error(f"Error computing χ² for concentration {conc}: {e}")
                chi_squared_values.append(1e12)  # Penalty for failed computation
        
        result = np.array(chi_squared_values)
        return pt.as_tensor(result, dtype='float64') if isinstance(kd_value, pt.TensorVariable) else result
    
    def _compute_single_chi_squared(self, exp_file: Path, concentration: float, kd: float) -> float:
        """Compute chi-squared for a single concentration."""
        M, O = self.equilibrium_solver.solve_system(concentration, kd)
        
        if np.isnan(M):
            return 1e12
        
        monomer_fraction = M / concentration
        oligomer_fraction = self.n_value * O / concentration
        
        # Calculate theoretical intensity
        theoretical_sum = self.data_handler.calculate_theoretical_intensity(
            monomer_fraction, oligomer_fraction)
        
        # Save files and run ATSAS
        theoretical_file = Path(OUTPUT_DIRS['theoretical']) / f"theoretical_{kd}_{concentration}.int"
        fit_file = Path(OUTPUT_DIRS['fits']) / f"fit_{concentration}_{kd}.fit"
        log_file = Path(OUTPUT_DIRS['logs']) / f"oligomer_{concentration}_{kd}.log"
        
        np.savetxt(theoretical_file, theoretical_sum)
        
        self._run_atsas_oligomer(theoretical_file, exp_file, fit_file, log_file)
        
        chi_squared = self._extract_chi_squared(log_file)
        return self._apply_chi_squared_penalties(chi_squared)
    
    def _run_atsas_oligomer(self, theo_file: Path, exp_file: Path, 
                           fit_file: Path, log_file: Path) -> None:
        """Run ATSAS oligomer software for SAXS analysis."""
        cmd = [
            str(ATSAS_PATH / "oligomer"),
            "-ff", str(theo_file),
            str(exp_file),
            f"--fit={fit_file}",
            f"--out={log_file}",
            "-cst", "-ws", f"-un={self.n_value}"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ATSAS oligomer failed: {e}")
            raise
    
    @staticmethod
    def _extract_chi_squared(log_file: Path) -> Optional[float]:
        """Extract chi-squared value from ATSAS log file."""
        try:
            with open(log_file, 'r') as file:
                log_content = file.read()
            matches = re.findall(r'\.dat.*?(\d+\.\d+)', log_content)
            return float(matches[1]) if len(matches) >= 2 else None
        except Exception as e:
            logger.error(f"Error extracting χ² from log file: {e}")
            return None
    
    @staticmethod
    def _apply_chi_squared_penalties(chi_squared: Optional[float]) -> float:
        """Apply penalties to chi-squared values based on fit quality."""
        if chi_squared is None:
            return 1e12
        
        if chi_squared > 5:  # Poor fit
            return chi_squared * 10 + (chi_squared - 5)**3
        elif chi_squared > 2:  # Marginal fit
            return chi_squared * 5 + (chi_squared - 2)**2
        elif chi_squared > 1.5:  # Acceptable fit
            return chi_squared * 2
        return chi_squared  # Good fit
    
    def run_mcmc(self) -> az.InferenceData:
        """Run MCMC sampling for Kd determination."""
        logger.info("Starting MCMC sampling...")
        
        with pm.Model() as model:
            # Prior: log-uniform for Kd
            log_kd = pm.Uniform('log_kd', 
                              lower=np.float64(np.log10(self.kd_range[0])), 
                              upper=np.float64(np.log10(self.kd_range[1])))
            
            # Transform to actual Kd
            kd = pm.Deterministic('kd', 10**log_kd)
            
            # Likelihood
            chi_squared = self.compute_chi_squared_for_kd(kd)
            likelihood = -0.5 * pt.sum(chi_squared)
            pm.Potential('likelihood', likelihood)
            
            # Run MCMC
            try:
                self.trace = pm.sample(
                    draws=self.mcmc_params['samples'],
                    tune=self.mcmc_params['tune'],
                    chains=self.mcmc_params['chains'],
                    return_inferencedata=True,
                    target_accept=self.mcmc_params['target_accept'],
                    init='adapt_diag'
                )
            except Exception as e:
                logger.error(f"MCMC sampling failed: {e}")
                raise
        
        self.kd_samples = self.trace.posterior['kd'].values.flatten()
        logger.info("MCMC sampling completed successfully")
        
        return self.trace
    
    def get_results_summary(self) -> Dict[str, float]:
        """Get summary statistics of MCMC results."""
        if self.kd_samples is None:
            raise ValueError("No MCMC results available. Run run_mcmc() first.")
        
        return {
            'mean_kd': float(np.mean(self.kd_samples)),
            'std_kd': float(np.std(self.kd_samples)),
            'ci_lower': float(np.percentile(self.kd_samples, 2.5)),
            'ci_upper': float(np.percentile(self.kd_samples, 97.5))
        }
    
    def find_good_kd_range(self, all_results: pd.DataFrame) -> List[float]:
        """Find Kd values where all concentrations have χ² < threshold."""
        good_kds = []
        for kd in all_results['kd'].unique():
            kd_results = all_results[all_results['kd'] == kd]
            if all(kd_results['chi2'] < CHI2_THRESHOLD):
                good_kds.append(kd)
        return good_kds
    
    def calculate_chi_squared(self, exp_file: Path, concentration: float) -> pd.DataFrame:
        """
        Calculate chi-squared values using ATSAS oligomer across Kd range.
        
        Parameters
        ----------
        exp_file : Path
            Path to experimental data file
        concentration : float
            Protein concentration in µM
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - kd: Kd values
            - concentration: protein concentration
            - mon_frac: monomer fraction
            - dim_frac: dimer fraction
            - chi2: chi-squared values from ATSAS oligomer
            
        Notes
        -----
        Uses ATSAS oligomer to calculate chi-squared values for each Kd.
        Includes penalties for poor fits based on chi-squared thresholds.
        """
        try:
            # Create Kd values array
            kd_values = np.round(np.geomspace(self.kd_range[0], self.kd_range[1], 
                                             num=KD_POINTS), decimals=2)
            chi_squared_values = []
            
            for kd in kd_values:
                # Calculate fractions using equilibrium solver
                M, O = self.equilibrium_solver.solve_system(concentration, kd)
                
                if not np.isnan(M):
                    monomer_fraction = M / concentration
                    oligomer_fraction = self.n_value * O / concentration
                    
                    # Calculate theoretical intensity
                    theoretical_sum = self.data_handler.calculate_theoretical_intensity(
                        monomer_fraction, oligomer_fraction)
                    
                    # Save theoretical data
                    theoretical_file = Path(OUTPUT_DIRS['theoretical']) / f"theoretical_{kd}.int"
                    np.savetxt(theoretical_file, theoretical_sum)
                    
                    # Set up ATSAS files
                    fit_file = Path(OUTPUT_DIRS['fits']) / f"fit_{concentration}_{kd}.fit"
                    log_file = Path(OUTPUT_DIRS['logs']) / f"oligomer_{concentration}_{kd}.log"
                    
                    self._run_atsas_oligomer(theoretical_file, exp_file, fit_file, log_file)
                    
                    # Extract chi-squared from log file
                    chi_squared = self._extract_chi_squared(log_file)
                    chi_squared_values.append((kd, concentration, monomer_fraction, 
                                            oligomer_fraction, chi_squared))
            
            return pd.DataFrame(chi_squared_values, 
                              columns=["kd", "concentration", "mon_frac", "dim_frac", "chi2"])
        
        except Exception as e:
            logger.error(f"Error in calculation: {e}")
            raise

class SAXSPlotter:
    """
    Visualization class for SAXS analysis results.
    
    Generates publication-quality plots of SAXS analysis results including
    chi-squared analysis, molecular fractions, and MCMC diagnostics.
    """
    
    def __init__(self, results: List[pd.DataFrame], trace: az.InferenceData, 
                 output_dir: Path = Path(OUTPUT_DIRS['figures'])):
        """Initialize plotter with analysis results."""
        self.results = results
        self.trace = trace
        self.output_dir = output_dir
        self.kd_samples = trace.posterior['kd'].values.flatten()
        
        # Calculate statistics
        self.kd_mean = float(np.mean(self.kd_samples))
        self.kd_std = float(np.std(self.kd_samples))
        self.kd_interval = np.percentile(self.kd_samples, [2.5, 97.5])
        
        # Initialize figures
        self.fig_main = None
        self.fig_diag = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def create_main_figure(self) -> go.Figure:
        """Create main results figure with four subplots."""
        self.fig_main = make_subplots(
            rows=1, cols=4,
            subplot_titles=("χ² vs Kd", "χ² < 2 Region", 
                          "Molecular Fractions", "Kd Posterior"),
            horizontal_spacing=0.08,
            specs=[[{"type": "scatter"}, {"type": "scatter"}, 
                   {"type": "scatter"}, {"type": "histogram"}]]
        )
        
        self._add_chi_squared_plot()
        self._add_zoomed_chi_squared_plot()
        self._add_molecular_fractions_plot()
        self._add_posterior_plot()
        self._update_main_layout()
        
        # Show the figure
        self.fig_main.show()
        
        return self.fig_main
    
    def create_diagnostic_figure(self) -> go.Figure:
        """Create MCMC diagnostics figure."""
        self.fig_diag = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Trace Plot", "Chain Distributions"),
            specs=[[{"type": "scatter"}, {"type": "violin"}]]
        )
        
        self._add_trace_plots()
        self._add_chain_distribution()
        self._update_diagnostic_layout()
        
        # Show the figure
        self.fig_diag.show()
        
        return self.fig_diag
    
    def _add_chi_squared_plot(self) -> None:
        """Add chi-squared vs Kd plot."""
        chi_squared_values = pd.concat(self.results)
        avg_chi_squared = chi_squared_values.groupby('kd')['chi2'].mean().reset_index()
        
        for concentration in chi_squared_values['concentration'].unique():
            df_subset = chi_squared_values[chi_squared_values['concentration'] == concentration]
            self.fig_main.add_trace(
                go.Scatter(x=df_subset['kd'], y=df_subset['chi2'],
                          mode='lines+markers', name=f'{concentration} µM'),
                row=1, col=1
            )
        
        self.fig_main.add_trace(
            go.Scatter(x=avg_chi_squared['kd'], y=avg_chi_squared['chi2'],
                      mode='lines+markers', name='Average',
                      line=dict(color='black', width=2, dash='dash')),
            row=1, col=1
        )
    
    def _add_molecular_fractions_plot(self) -> None:
        """Add molecular fractions plot with uncertainty bands."""
        concentration_range = np.logspace(np.log10(CONC_RANGE[0]), 
                                        np.log10(CONC_RANGE[1]), 
                                        CONC_POINTS)
        
        equilibrium_solver = EquilibriumSolver(N_VALUE)
        mon_fracs = []
        oligo_fracs = []
        
        for kd in self.kd_samples[::10]:  # Thin samples for efficiency
            fractions = equilibrium_solver.calculate_fractions(kd, concentration_range)
            mon_fracs.append(fractions['monomer_fraction'].values)
            oligo_fracs.append(fractions['oligomer_fraction'].values)
        
        mon_fracs = np.array(mon_fracs)
        oligo_fracs = np.array(oligo_fracs)
        
        self._plot_fraction_with_uncertainty(concentration_range, mon_fracs, 
                                           'Monomer', 'green', 3)
        self._plot_fraction_with_uncertainty(concentration_range, oligo_fracs, 
                                           'Oligomer', 'red', 3)
    
    def _plot_fraction_with_uncertainty(self, conc_range: np.ndarray, 
                                      fracs: np.ndarray, name: str, 
                                      color: str, col: int) -> None:
        """
        Plot molecular fraction with uncertainty band.
        
        Parameters
        ----------
        conc_range : np.ndarray
            Concentration range for x-axis
        fracs : np.ndarray
            Array of fraction values
        name : str
            Name of the species (Monomer/Oligomer)
        color : str
            Color for the plot (e.g., 'green', 'red')
        col : int
            Column index in the subplot
        """
        mean = np.mean(fracs, axis=0)
        lower, upper = np.percentile(fracs, [2.5, 97.5], axis=0)
        
        # Main line
        self.fig_main.add_trace(
            go.Scatter(
                x=conc_range, 
                y=mean, 
                mode='lines', 
                name=name, 
                line=dict(color=color)
            ),
            row=1, col=col
        )
        
        # Uncertainty band
        if color == 'green':
            fill_color = 'rgba(0,255,0,0.2)'
        elif color == 'red':
            fill_color = 'rgba(255,0,0,0.2)'
        else:
            fill_color = 'rgba(128,128,128,0.2)'
        
        self.fig_main.add_trace(
            go.Scatter(
                x=np.concatenate([conc_range, conc_range[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself',
                fillcolor=fill_color,
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{name} 95% CI'
            ),
            row=1, col=col
        )
    
    def save_figures(self, format_list: List[str] = ['html', 'png']) -> None:
        """Save figures in specified formats."""
        for fmt in format_list:
            try:
                if fmt == 'html':
                    self.fig_main.write_html(self.output_dir / "saxs_analysis_results.html")
                    self.fig_diag.write_html(self.output_dir / "saxs_diagnostics.html")
                else:
                    self.fig_main.write_image(self.output_dir / f"saxs_analysis_results.{fmt}")
                    self.fig_diag.write_image(self.output_dir / f"saxs_diagnostics.{fmt}")
            except Exception as e:
                logger.error(f"Error saving figures in {fmt} format: {e}")
    
    
    def _add_zoomed_chi_squared_plot(self) -> None:
        """Add zoomed chi-squared plot (χ² < 2 region) to main figure."""
        chi_squared_values = pd.concat(self.results)
        avg_chi_squared = chi_squared_values.groupby('kd')['chi2'].mean().reset_index()
        
        # Add χ² = 2 threshold line
        self.fig_main.add_hline(
            y=2, 
            line_dash="dash", 
            line_color="red",
            row=1, col=2, 
            annotation_text="χ² = 2"
        )
        
        # Plot individual concentration curves
        for concentration in chi_squared_values['concentration'].unique():
            df_subset = chi_squared_values[chi_squared_values['concentration'] == concentration]
            self.fig_main.add_trace(
                go.Scatter(
                    x=df_subset['kd'], 
                    y=df_subset['chi2'],
                    mode='lines+markers',
                    name=f'{concentration} µM',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add average curve
        self.fig_main.add_trace(
            go.Scatter(
                x=avg_chi_squared['kd'], 
                y=avg_chi_squared['chi2'],
                mode='lines+markers', 
                name='Average',
                line=dict(color='black', width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
    
    def _add_posterior_plot(self) -> None:
        """Add Kd posterior distribution histogram."""
        self.fig_main.add_trace(
            go.Histogram(
                x=self.kd_samples,
                nbinsx=50,
                name='Kd posterior'
            ),
            row=1, col=4
        )
    
    def _update_main_layout(self) -> None:
        """Update layout of main figure."""
        self.fig_main.update_xaxes(type="log", title="Kd (µM)", row=1, col=1)
        self.fig_main.update_xaxes(type="log", title="Kd (µM)", row=1, col=2)
        self.fig_main.update_xaxes(type="log", title="Concentration (µM)", row=1, col=3)
        self.fig_main.update_xaxes(title="Kd (µM)", row=1, col=4)
        
        self.fig_main.update_yaxes(title="χ²", row=1, col=1)
        self.fig_main.update_yaxes(title="χ²", range=[0, 2], row=1, col=2)  # Set y-axis range for zoomed plot
        self.fig_main.update_yaxes(title="Fraction", range=[0, 1], row=1, col=3)
        self.fig_main.update_yaxes(title="Frequency", row=1, col=4)
        
        self.fig_main.update_layout(
            height=500,
            width=2000,
            title_text=f"SAXS Analysis Results (Kd = {self.kd_mean:.2f} ± {self.kd_std:.2f} µM)",
            template='simple_white'
        )
    
    def _add_trace_plots(self) -> None:
        """Add MCMC trace plots."""
        for chain in range(self.trace.posterior.dims['chain']):
            chain_data = self.trace.posterior['kd'].isel(chain=chain)
            self.fig_diag.add_trace(
                go.Scatter(
                    y=chain_data,
                    mode='lines',
                    name=f'Chain {chain}',
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    def _add_chain_distribution(self) -> None:
        """Add chain distribution violin plot."""
        self.fig_diag.add_trace(
            go.Violin(
                y=self.kd_samples,
                box_visible=True,
                meanline_visible=True,
                name='Posterior'
            ),
            row=1, col=2
        )
    
    def _update_diagnostic_layout(self) -> None:
        """Update layout of diagnostic figure."""
        self.fig_diag.update_layout(
            height=400,
            width=800,
            title_text="MCMC Diagnostics",
            template='simple_white'
        )

def main():
    """Main execution function."""
    logger.info("Starting SAXS analysis...")
    
    # Create output directories
    for dir_name in OUTPUT_DIRS.values():
        os.makedirs(dir_name, exist_ok=True)
    
    try:
        # Initialize data handler
        data_handler = SAXSDataHandler(THEO_MON_FILE, THEO_DIM_FILE)
        
        # Process experimental files
        exp_files_full = [Path(EXP_DIR) / f for f in EXP_FILES]
        
        # Initialize analysis
        analysis = BayesianSAXSAnalysis(
            experimental_files=exp_files_full,
            concentrations=CONCENTRATIONS,
            data_handler=data_handler
        )
        
        # Run initial chi-squared analysis
        results = []
        for conc, exp_file in zip(CONCENTRATIONS, exp_files_full):
            logger.info(f"Processing concentration {conc} µM...")
            result = analysis.calculate_chi_squared(exp_file, conc)
            results.append(result)
        
        # Find good Kd range
        all_results = pd.concat(results)
        good_kds = analysis.find_good_kd_range(all_results)
        
        if good_kds:
            good_kd_min = min(good_kds)
            good_kd_max = max(good_kds)
            logger.info(f"\nFound Kd range with χ² < 2 for all concentrations:")
            logger.info(f"Kd range: {good_kd_min:.1f} - {good_kd_max:.1f} µM")
            
            # Update the Kd range for MCMC - This is crucial!
            analysis.kd_range = (good_kd_min, good_kd_max)
            
            # Run Bayesian analysis with updated range
            logger.info("Running Bayesian analysis...")
            trace = analysis.run_mcmc()
            
            # Create plots
            plotter = SAXSPlotter(results=results, trace=trace)
            plotter.create_main_figure()
            plotter.create_diagnostic_figure()
            plotter.save_figures(['html', 'png'])
            
            # Print results
            summary = analysis.get_results_summary()
            logger.info("\nAnalysis Results:")
            logger.info(f"Kd = {summary['mean_kd']:.2f} ± {summary['std_kd']:.2f} µM")
            logger.info(f"95% Credible Interval: [{summary['ci_lower']:.2f}, {summary['ci_upper']:.2f}] µM")
            
        else:
            logger.warning("No Kd values found where all concentrations have χ² < 2")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    
    logger.info("Analysis completed successfully")

if __name__ == "__main__":
    main()

