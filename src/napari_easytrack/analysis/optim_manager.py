"""Optimization manager --- :mod:`napari_easytrack.analysis.optim_manager`
=========================================================================
Optimization manager for running btrack parameter optimization in background.

This module handles:
- Starting optimization in a separate thread
- Monitoring progress by polling Optuna study database
- Cancelling running optimizations
- Retrieving best trials after completion
"""

import threading
import time
from typing import Optional, Callable, Dict, List, Tuple

import optuna
from qtpy.QtCore import QObject, Signal

from .optim_pipeline import optimize_dataset_with_timeout


class OptimizationSignals(QObject):
    """Qt signals for thread-safe communication."""
    finished = Signal(object)  # Emits study object
    error = Signal(str)  # Emits error message


class OptimizationManager:
    """
    Manages background optimisation and progress monitoring.

    This class uses Optuna for optimisation and runs the process in a separate thread.

    Attributes:
        db_path: Path to SQLite database for Optuna studies
        db_url: SQLAlchemy database URL
        signals: Qt signals for communication
        current_thread: Thread object for running optimisation
        current_study_name: Name of the current study
        is_running: Whether an optimisation is currently running
        should_cancel: Flag to indicate cancellation request
        start_time: Timestamp when optimisation started
        study: Optuna Study object after completion

    Methods:
        start_optimization: Start optimisation in background thread
        get_progress: Get current progress of optimisation
        cancel_current: Cancel currently running optimisation
        is_complete: Check if optimisation has completed
        get_best_trials: Retrieve best trials from completed study
        study_exists: Check if a study exists in the database
        get_study_summary: Get summary statistics for a study
        cleanup: Clean up connections and state
    """
    
    def __init__(self, db_path: str = "btrack.db"):
        """
        Initialize optimization manager.
        
        Args:
            db_path: Path to SQLite database for Optuna studies
        """
        self.db_path = db_path
        self.db_url = f"sqlite:///{db_path}"
        
        # Qt signals for thread-safe communication
        self.signals = OptimizationSignals()
        
        # State tracking
        self.current_thread: Optional[threading.Thread] = None
        self.current_study_name: Optional[str] = None
        self.is_running = False
        self.should_cancel = False
        self.start_time: Optional[float] = None
        
        # Study object (set after optimization completes)
        self.study: Optional[optuna.Study] = None
        
        # Callbacks (will be stored but not called directly from thread)
        self._on_finished: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
    
    def start_optimization(
        self,
        dataset,
        gt_data,
        study_name: str,
        n_trials: int = 128,
        timeout: int = 60,
        timeout_penalty: float = 100000,
        sampler: str = 'tpe',
        use_parallel_backend: bool = True,
        on_progress: Optional[Callable] = None,
        on_finished: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Start optimization in background thread.
        
        Args:
            dataset: CellTrackingChallengeDataset object
            gt_data: Ground truth TrackingGraph
            study_name: Name for Optuna study
            n_trials: Number of optimization trials
            timeout: Timeout per trial in seconds
            timeout_penalty: Penalty value for timeouts
            sampler: Sampler type ('tpe' or 'random')
            use_parallel_backend: Whether to use parallel processing
            on_progress: Callback(trial_num, best_aogm, elapsed_time)
            on_finished: Callback(study) when complete
            on_error: Callback(error_msg) on failure
        """
        if self.is_running:
            raise RuntimeError("Optimization already running. Cancel it first.")
        
        # Reset state
        self.current_study_name = study_name
        self.is_running = True
        self.should_cancel = False
        self.start_time = time.time()
        self.study = None
        
        # Store callbacks
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_error = on_error
        
        # Connect signals to callbacks (thread-safe)
        if on_finished:
            self.signals.finished.connect(on_finished)
        if on_error:
            self.signals.error.connect(on_error)
        
        # Start optimization thread
        self.current_thread = threading.Thread(
            target=self._run_optimization,
            args=(
                dataset, 
                gt_data, 
                study_name, 
                n_trials, 
                timeout,
                timeout_penalty,
                sampler, 
                use_parallel_backend
            ),
            daemon=True
        )
        self.current_thread.start()
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION STARTED")
        print(f"{'='*60}")
        print(f"Study: {study_name}")
        print(f"Trials: {n_trials}")
        print(f"Timeout: {timeout}s per trial")
        print(f"Timeout Penalty: {timeout_penalty}")
        print(f"Sampler: {sampler}")
        print(f"Parallel: {use_parallel_backend}")
        print(f"{'='*60}\n")
    
    def _run_optimization(
        self, 
        dataset, 
        gt_data, 
        study_name, 
        n_trials, 
        timeout,
        timeout_penalty,
        sampler, 
        use_parallel_backend
    ):
        """
        Internal method to run optimization (runs in separate thread).
        """
        try:
            # Call your optimization function
            # Note: objectives is hardcoded to '1obj' since we're not handling divisions
            self.study = optimize_dataset_with_timeout(
                dataset=dataset,
                gt_data=gt_data,
                objectives='1obj',
                study_name=study_name,
                n_trials=n_trials,
                timeout=timeout,
                timeout_penalty=timeout_penalty,
                use_parallel_backend=use_parallel_backend,
                sampler=sampler
            )
            
            # Success - emit signal (thread-safe)
            self.is_running = False
            self.signals.finished.emit(self.study)
                
        except Exception as e:
            # Error occurred - emit signal (thread-safe)
            self.is_running = False
            error_msg = f"Optimization failed: {type(e).__name__}: {str(e)}"
            print(error_msg)
            self.signals.error.emit(error_msg)


    
    def get_progress(self) -> Optional[Tuple[int, float, int]]:
        """
        Get current optimization progress by querying study database.
        
        Returns:
            Tuple of (trial_number, best_aogm, elapsed_seconds) or None if no progress
        """
        if not self.is_running or not self.current_study_name:
            return None
        
        try:
            # Load study from database
            storage = optuna.storages.RDBStorage(
                url=self.db_url,
                engine_kwargs={"connect_args": {"timeout": 10}}
            )
            study = optuna.load_study(
                study_name=self.current_study_name,
                storage=storage
            )
            
            # Get completed trials
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if not completed_trials:
                return None
            
            # Get best trial so far
            trial_num = len(completed_trials)
            best_trial = min(completed_trials, key=lambda t: t.value if t.value is not None else float('inf'))
            best_aogm = best_trial.value if best_trial.value is not None else float('inf')
            
            # Calculate elapsed time
            elapsed = int(time.time() - self.start_time) if self.start_time else 0
            
            return trial_num, best_aogm, elapsed
            
        except Exception as e:
            # Study might not exist yet or database locked
            return None
    
    def cancel_current(self):
        """
        Cancel currently running optimization.
        
        Note: This is a soft cancel - the current trial will complete but no new trials will start.
        """
        if not self.is_running:
            return
        
        print("\n⚠️  Cancellation requested...")
        print("The current trial will complete, but no new trials will start.")
        self.should_cancel = True
        
        # Unfortunately, Optuna doesn't have a clean way to cancel mid-optimization
        # The best we can do is let the current trial finish and then the optimization
        # will stop naturally. The study will still be saved with completed trials.
    
    def is_complete(self) -> bool:
        """Check if optimization has completed."""
        return not self.is_running and self.current_thread is not None
    
    def get_best_trials(self, study_name: Optional[str] = None, max_trials: int = 15) -> List[Dict]:
        """
        Get best trials from a completed study.
        
        Args:
            study_name: Name of study to load (uses current if None)
            max_trials: Maximum number of best trials to return
            
        Returns:
            List of dicts containing trial info (params, AOGM, number)
        """
        if study_name is None:
            study_name = self.current_study_name
        
        if study_name is None:
            raise ValueError("No study name provided and no current study")
        
        try:
            # Load study from database
            storage = optuna.storages.RDBStorage(
                url=self.db_url,
                engine_kwargs={"connect_args": {"timeout": 10}}
            )
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )
            
            # Get completed trials sorted by objective value
            completed_trials = [
                t for t in study.trials 
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
            ]
            
            if not completed_trials:
                return []
            
            # Sort by AOGM (lower is better)
            sorted_trials = sorted(completed_trials, key=lambda t: t.value)
            
            # Get best trials (handle ties by taking top max_trials)
            best_value = sorted_trials[0].value
            best_trials = []
            
            for trial in sorted_trials[:max_trials]:
                # Include trials that are very close to the best (within 1% tolerance for ties)
                if abs(trial.value - best_value) / (best_value + 1e-10) < 0.01:
                    best_trials.append({
                        'number': trial.number,
                        'aogm': trial.value,
                        'params': trial.params,
                        'user_attrs': trial.user_attrs  # Contains other metrics like TRA, DET, etc.
                    })
                else:
                    # If we've included at least one trial and this one isn't a tie, we can stop
                    if best_trials:
                        break
                    # Otherwise include this as the best so far
                    best_trials.append({
                        'number': trial.number,
                        'aogm': trial.value,
                        'params': trial.params,
                        'user_attrs': trial.user_attrs
                    })
            
            return best_trials
            
        except Exception as e:
            print(f"Error loading best trials: {e}")
            return []
    
    def study_exists(self, study_name: str) -> bool:
        """
        Check if a study with given name exists in database.
        
        Args:
            study_name: Name of study to check
            
        Returns:
            True if study exists, False otherwise
        """
        try:
            storage = optuna.storages.RDBStorage(
                url=self.db_url,
                engine_kwargs={"connect_args": {"timeout": 10}}
            )
            study_names = storage.get_all_study_names()
            return study_name in study_names
        except Exception:
            return False
    
    def get_study_summary(self, study_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get summary statistics for a study.
        
        Args:
            study_name: Name of study (uses current if None)
            
        Returns:
            Dict with study summary or None if study not found
        """
        if study_name is None:
            study_name = self.current_study_name
        
        if study_name is None:
            return None
        
        try:
            storage = optuna.storages.RDBStorage(
                url=self.db_url,
                engine_kwargs={"connect_args": {"timeout": 10}}
            )
            study = optuna.load_study(
                study_name=study_name,
                storage=storage
            )
            
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            
            if not completed:
                return {
                    'total_trials': len(study.trials),
                    'completed_trials': 0,
                    'failed_trials': len(failed),
                    'best_aogm': None
                }
            
            best_trial = min(completed, key=lambda t: t.value if t.value is not None else float('inf'))
            
            return {
                'study_name': study_name,
                'total_trials': len(study.trials),
                'completed_trials': len(completed),
                'failed_trials': len(failed),
                'best_aogm': best_trial.value,
                'best_trial_number': best_trial.number,
                'sampler': study.sampler.__class__.__name__
            }
            
        except Exception as e:
            print(f"Error getting study summary: {e}")
            return None
    
    def cleanup(self):
        """Clean up connections and state."""
        try:
            self.signals.finished.disconnect()
        except:
            pass
        try:
            self.signals.error.disconnect()
        except:
            pass
        
        self._on_finished = None
        self._on_error = None
        self._on_progress = None

        # Force close any Optuna SQLite connections
        # This is critical on Windows where file locks persist
        if hasattr(self, 'study') and self.study is not None:
            try:
                # Close the study's storage connection
                if hasattr(self.study._storage, '_engine'):
                    self.study._storage._engine.dispose()
            except:
                pass
            self.study = None

        # Force garbage collection to release any lingering database connections
        import gc
        gc.collect()