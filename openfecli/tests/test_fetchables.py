import pytest
from click.testing import CliRunner
from .conftest import HAS_INTERNET
import pathlib

from openfecli.fetching import FetchablePlugin

from openfecli.fetchables import (
    RHFE_TUTORIAL, RHFE_TUTORIAL_RESULTS
)

def fetchable_test(fetchable):
    """Unit test to ensure that a given FetchablePlugin works"""
    assert isinstance(fetchable, FetchablePlugin)
    expected_paths = [pathlib.Path(f) for f in fetchable.filenames]
    runner = CliRunner()
    if fetchable.fetcher.REQUIRES_INTERNET and not HAS_INTERNET:  # -no-cov-
        pytest.skip("Internet seems to be unavailable")
    with runner.isolated_filesystem():
        result = runner.invoke(fetchable.command, ['-d' 'output-dir'])
        assert result.exit_code == 0
        for path in expected_paths:
            assert (pathlib.Path("output-dir") / path).exists()


def test_rhfe_tutorial():
    fetchable_test(RHFE_TUTORIAL)

def test_rhfe_tutorial_results():
    fetchable_test(RHFE_TUTORIAL_RESULTS)
