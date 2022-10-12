"""
Provides the context for pytest. The relative import is necessary to test the module
without requiring an installation by the user.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from pathlib import Path
from experiments import *

@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    """
    Creates a temporary data directory with pseudo-data for tests.

    Returns
    -------
    data_path : pathlib Path
        Parent of temporary data directory.
    """

    # Configure tmp data path
    data_path = tmp_path_factory.mktemp("data")
    assert isinstance(data_path, Path)

    # Implement data structure
    # file_list = ["point.dat", "line.dat", "cadastre.dat"]
    # for file in file_list:
    #     (data_path / file).touch()
    return data_path
