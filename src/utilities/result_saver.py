# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper module for saving optimization results to HDF5 files."""

import os
import h5py
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import asdict, is_dataclass


class ResultSaver:
    """
    A helper class to save optimization results to HDF5 files.
    
    Each optimizer class gets its own master file in results/ directory,
    and each run creates a timestamped file in tmp/ directory.
    """
    
    def __init__(self, optimizer_name: str, base_dir: str = "."):
        """
        Initialize the ResultSaver.
        
        Parameters:
        -----------
        optimizer_name : str
            Name of the optimizer class (e.g., 'toriccode', 'gibbs_fidelity')
        base_dir : str
            Base directory for results (default: current directory)
        """
        self.optimizer_name = optimizer_name.lower()
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, "results")
        self.tmp_dir = os.path.join(base_dir, "tmp")
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Master file path
        self.master_file = os.path.join(self.results_dir, f"{self.optimizer_name}.h5")
    
    def _serialize_settings(self, settings: Any) -> Dict:
        """
        Convert settings object to a dictionary.
        
        Parameters:
        -----------
        settings : Any
            Settings object (dataclass, dict, or object with __dict__)
            
        Returns:
        --------
        dict : Dictionary representation of settings
        """
        if is_dataclass(settings):
            return asdict(settings)
        elif isinstance(settings, dict):
            return settings
        elif hasattr(settings, '__dict__'):
            return settings.__dict__
        else:
            return {'settings': str(settings)}
    
    def _save_to_hdf5(self, filepath: str, settings: Dict, results: Dict, 
                      append: bool = False) -> None:
        """
        Save data to HDF5 file.
        
        Parameters:
        -----------
        filepath : str
            Path to HDF5 file
        settings : dict
            Dictionary of settings/parameters
        results : dict
            Dictionary of optimization results
        append : bool
            If True, append to existing file; if False, create new file
        """
        mode = 'a' if append else 'w'
        
        with h5py.File(filepath, mode) as f:
            # Generate unique run ID based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_group = f.create_group(f"run_{timestamp}")
            
            # Save settings
            settings_group = run_group.create_group("settings")
            for key, value in settings.items():
                if value is None:
                    settings_group.attrs[key] = "None"
                elif isinstance(value, (list, tuple)):
                    settings_group.create_dataset(key, data=np.array(value))
                elif isinstance(value, np.ndarray):
                    settings_group.create_dataset(key, data=value)
                elif isinstance(value, (int, float, str, bool)):
                    settings_group.attrs[key] = value
                else:
                    # For complex objects, store as string
                    settings_group.attrs[key] = str(value)
            
            # Save results
            results_group = run_group.create_group("results")
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results_group.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    results_group.create_dataset(key, data=np.array(value))
                elif isinstance(value, (int, float, str, bool)):
                    results_group.attrs[key] = value
                else:
                    # For complex objects, try to convert to array or store as string
                    try:
                        results_group.create_dataset(key, data=np.array(value))
                    except:
                        results_group.attrs[key] = str(value)
            
            # Add metadata
            run_group.attrs['timestamp'] = timestamp
            run_group.attrs['optimizer'] = self.optimizer_name
    
    def save_run(self, settings: Any, results: Dict, 
                 save_individual: bool = True) -> tuple[str, Optional[str]]:
        """
        Save a single optimization run.
        
        This method:
        1. Appends the run to the master file in results/
        2. Optionally saves a timestamped individual file in tmp/
        
        Parameters:
        -----------
        settings : Any
            Settings/parameters used for this run (dataclass, dict, or object)
        results : dict
            Results from the optimization run. Should contain keys like:
            - 'final_energies': final energy values
            - 'final_parameters': final parameter values
            - 'all_energies': energy trajectory
            - 'all_purities': purity trajectory (if applicable)
        save_individual : bool
            If True, also save individual timestamped file in tmp/
            
        Returns:
        --------
        tuple : (master_file_path, individual_file_path or None)
        """
        # Serialize settings
        settings_dict = self._serialize_settings(settings)
        
        # Save to master file (append mode)
        self._save_to_hdf5(self.master_file, settings_dict, results, append=True)
        
        individual_file = None
        if save_individual:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            individual_file = os.path.join(
                self.tmp_dir, 
                f"{self.optimizer_name}_{timestamp}.h5"
            )
            # Save individual file (new file)
            self._save_to_hdf5(individual_file, settings_dict, results, append=False)
        
        return self.master_file, individual_file
    
    @staticmethod
    def load_run(filepath: str, run_id: Optional[str] = None) -> Dict:
        """
        Load a run from an HDF5 file.
        
        Parameters:
        -----------
        filepath : str
            Path to HDF5 file
        run_id : str, optional
            Specific run ID to load. If None, loads the most recent run.
            
        Returns:
        --------
        dict : Dictionary containing 'settings' and 'results'
        """
        with h5py.File(filepath, 'r') as f:
            # Get run IDs
            run_ids = [key for key in f.keys() if key.startswith('run_')]
            
            if not run_ids:
                raise ValueError(f"No runs found in {filepath}")
            
            # Select run
            if run_id is None:
                # Get most recent run (last in sorted list)
                run_id = sorted(run_ids)[-1]
            elif run_id not in run_ids:
                raise ValueError(f"Run {run_id} not found. Available runs: {run_ids}")
            
            run_group = f[run_id]
            
            # Load settings
            settings = {}
            settings_group = run_group['settings']
            for key in settings_group.attrs:
                settings[key] = settings_group.attrs[key]
            for key in settings_group.keys():
                settings[key] = settings_group[key][:]
            
            # Load results
            results = {}
            results_group = run_group['results']
            for key in results_group.attrs:
                results[key] = results_group.attrs[key]
            for key in results_group.keys():
                results[key] = results_group[key][:]
            
            # Add metadata
            metadata = {
                'timestamp': run_group.attrs.get('timestamp', 'unknown'),
                'optimizer': run_group.attrs.get('optimizer', 'unknown')
            }
            
            return {
                'settings': settings,
                'results': results,
                'metadata': metadata
            }
    
    @staticmethod
    def list_runs(filepath: str) -> list:
        """
        List all runs in an HDF5 file.
        
        Parameters:
        -----------
        filepath : str
            Path to HDF5 file
            
        Returns:
        --------
        list : List of run IDs with their timestamps
        """
        with h5py.File(filepath, 'r') as f:
            runs = []
            for key in f.keys():
                if key.startswith('run_'):
                    timestamp = f[key].attrs.get('timestamp', 'unknown')
                    runs.append({'run_id': key, 'timestamp': timestamp})
            return sorted(runs, key=lambda x: x['timestamp'])

# Made with Bob
