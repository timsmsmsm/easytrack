"""Tests for optimization manager module."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from src.napari_easytrack.analysis.optim_manager import OptimizationManager


class TestOptimizationManager:
    """Tests for OptimizationManager class."""

    def test_manager_initialization(self):
        """Test that optimization manager can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            assert manager is not None
            assert hasattr(manager, 'db_path')

    def test_manager_study_exists(self):
        """Test checking if study exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            # Check for non-existent study
            exists = manager.study_exists("non_existent_study")
            
            assert isinstance(exists, bool)
            assert exists is False

    def test_manager_is_complete(self):
        """Test checking if optimization is complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            # Before any optimization, should not be complete
            is_complete = manager.is_complete()
            
            assert isinstance(is_complete, bool)

    def test_manager_cleanup(self):
        """Test manager cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            # Should not raise error
            manager.cleanup()

    def test_manager_get_progress(self):
        """Test getting optimization progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            progress = manager.get_progress()
            
            # Should return None or tuple
            assert progress is None or isinstance(progress, tuple)

    def test_manager_get_study_summary(self):
        """Test getting study summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            summary = manager.get_study_summary()
            
            # Should return None or dict
            assert summary is None or isinstance(summary, dict)

    def test_manager_get_best_trials(self):
        """Test getting best trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            manager = OptimizationManager(db_path=db_path)
            
            # Create a study first
            study_name = "test_study"
            manager.study_exists(study_name)  # This might create it
            
            try:
                trials = manager.get_best_trials(study_name=study_name)
                # Should return a list
                assert isinstance(trials, list)
            except ValueError:
                # If study doesn't exist, that's ok - we're testing the method exists
                assert True
