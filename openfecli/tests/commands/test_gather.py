from click.testing import CliRunner
from importlib import resources
import tarfile
import os
import pathlib
import pytest
import pooch

from openfecli.commands.gather import (
    gather, format_estimate_uncertainty, _get_column,
    _generate_bad_legs_error_message,
)

@pytest.mark.parametrize('est,unc,unc_prec,est_str,unc_str', [
    (12.432, 0.111, 2, "12.43", "0.11"),
    (0.9999, 0.01, 2, "1.000", "0.010"),
    (1234, 100, 2, "1230", "100"),
])
def test_format_estimate_uncertainty(est, unc, unc_prec, est_str, unc_str):
    assert format_estimate_uncertainty(est, unc, unc_prec) == (est_str, unc_str)

@pytest.mark.parametrize('val, col', [
    (1.0, 1), (0.1, -1), (-0.0, 0), (0.0, 0), (0.2, -1), (0.9, -1),
    (0.011, -2), (9, 1), (10, 2), (15, 2),
])
def test_get_column(val, col):
    assert _get_column(val) == col


@pytest.fixture
def results_dir(tmpdir):
    with tmpdir.as_cwd():
        with resources.files('openfecli.tests.data') as d:
            t = tarfile.open(d / 'rbfe_results.tar.gz', mode='r')
            t.extractall('.')

        yield

_EXPECTED_DG = b"""
ligand	DG(MLE) (kcal/mol)	uncertainty (kcal/mol)
lig_ejm_31	-0.09	0.05
lig_ejm_42	0.7	0.1
lig_ejm_46	-0.98	0.05
lig_ejm_47	-0.1	0.1
lig_ejm_48	0.53	0.09
lig_ejm_50	0.91	0.06
lig_ejm_43	2.0	0.2
lig_jmc_23	-0.68	0.09
lig_jmc_27	-1.1	0.1
lig_jmc_28	-1.25	0.08
"""

_EXPECTED_DDG = b"""
ligand_i	ligand_j	DDG(i->j) (kcal/mol)	uncertainty (kcal/mol)
lig_ejm_31	lig_ejm_42	0.8	0.1
lig_ejm_31	lig_ejm_46	-0.89	0.06
lig_ejm_31	lig_ejm_47	0.0	0.1
lig_ejm_31	lig_ejm_48	0.61	0.09
lig_ejm_31	lig_ejm_50	1.00	0.04
lig_ejm_42	lig_ejm_43	1.4	0.2
lig_ejm_46	lig_jmc_23	0.29	0.09
lig_ejm_46	lig_jmc_27	-0.1	0.1
lig_ejm_46	lig_jmc_28	-0.27	0.06
"""

_EXPECTED_DG_RAW = b"""
leg	ligand_i	ligand_j	DG(i->j) (kcal/mol)	uncertainty (kcal/mol)
complex	lig_ejm_31	lig_ejm_42	-15.0	0.1
solvent	lig_ejm_31	lig_ejm_42	-15.71	0.03
complex	lig_ejm_31	lig_ejm_46	-40.75	0.04
solvent	lig_ejm_31	lig_ejm_46	-39.86	0.05
complex	lig_ejm_31	lig_ejm_47	-27.8	0.1
solvent	lig_ejm_31	lig_ejm_47	-27.83	0.06
complex	lig_ejm_31	lig_ejm_48	-16.14	0.08
solvent	lig_ejm_31	lig_ejm_48	-16.76	0.03
complex	lig_ejm_31	lig_ejm_50	-57.33	0.04
solvent	lig_ejm_31	lig_ejm_50	-58.33	0.02
complex	lig_ejm_42	lig_ejm_43	-18.9	0.2
solvent	lig_ejm_42	lig_ejm_43	-20.28	0.03
complex	lig_ejm_46	lig_jmc_23	17.42	0.06
solvent	lig_ejm_46	lig_jmc_23	17.12	0.06
complex	lig_ejm_46	lig_jmc_27	15.81	0.09
solvent	lig_ejm_46	lig_jmc_27	15.91	0.05
complex	lig_ejm_46	lig_jmc_28	23.14	0.04
solvent	lig_ejm_46	lig_jmc_28	23.41	0.05
"""


_EXPECTED_RAW = b"""\
leg	ligand_i	ligand_j	DG(i->j) (kcal/mol)	MBAR uncertainty (kcal/mol)
complex	lig_ejm_31	lig_ejm_42	-14.77	0.04
complex	lig_ejm_31	lig_ejm_42	-14.74	0.04
complex	lig_ejm_31	lig_ejm_42	-14.94	0.04
solvent	lig_ejm_31	lig_ejm_42	-15.68	0.03
solvent	lig_ejm_31	lig_ejm_42	-15.69	0.03
solvent	lig_ejm_31	lig_ejm_42	-15.64	0.03
complex	lig_ejm_31	lig_ejm_46	-40.56	0.06
complex	lig_ejm_31	lig_ejm_46	-40.76	0.05
complex	lig_ejm_31	lig_ejm_46	-40.90	0.04
solvent	lig_ejm_31	lig_ejm_46	-39.92	0.04
solvent	lig_ejm_31	lig_ejm_46	-39.94	0.04
solvent	lig_ejm_31	lig_ejm_46	-39.95	0.04
complex	lig_ejm_31	lig_ejm_47	-27.68	0.08
complex	lig_ejm_31	lig_ejm_47	-27.80	0.06
complex	lig_ejm_31	lig_ejm_47	-27.51	0.07
solvent	lig_ejm_31	lig_ejm_47	-27.83	0.05
solvent	lig_ejm_31	lig_ejm_47	-27.84	0.05
solvent	lig_ejm_31	lig_ejm_47	-27.88	0.05
complex	lig_ejm_31	lig_ejm_48	-16.15	0.08
complex	lig_ejm_31	lig_ejm_48	-15.96	0.07
complex	lig_ejm_31	lig_ejm_48	-16.01	0.08
solvent	lig_ejm_31	lig_ejm_48	-16.83	0.06
solvent	lig_ejm_31	lig_ejm_48	-16.65	0.07
solvent	lig_ejm_31	lig_ejm_48	-16.77	0.06
complex	lig_ejm_31	lig_ejm_50	-57.31	0.04
complex	lig_ejm_31	lig_ejm_50	-57.45	0.04
complex	lig_ejm_31	lig_ejm_50	-57.37	0.04
solvent	lig_ejm_31	lig_ejm_50	-58.33	0.04
solvent	lig_ejm_31	lig_ejm_50	-58.42	0.04
solvent	lig_ejm_31	lig_ejm_50	-58.19	0.04
complex	lig_ejm_42	lig_ejm_43	-19.24	0.04
complex	lig_ejm_42	lig_ejm_43	-18.72	0.05
complex	lig_ejm_42	lig_ejm_43	-18.94	0.04
solvent	lig_ejm_42	lig_ejm_43	-20.17	0.03
solvent	lig_ejm_42	lig_ejm_43	-20.28	0.03
solvent	lig_ejm_42	lig_ejm_43	-20.23	0.03
complex	lig_ejm_46	lig_jmc_23	17.31	0.02
complex	lig_ejm_46	lig_jmc_23	17.37	0.02
complex	lig_ejm_46	lig_jmc_23	17.35	0.02
solvent	lig_ejm_46	lig_jmc_23	17.20	0.02
solvent	lig_ejm_46	lig_jmc_23	17.40	0.02
solvent	lig_ejm_46	lig_jmc_23	17.30	0.02
complex	lig_ejm_46	lig_jmc_27	15.84	0.03
complex	lig_ejm_46	lig_jmc_27	15.79	0.03
complex	lig_ejm_46	lig_jmc_27	15.80	0.03
solvent	lig_ejm_46	lig_jmc_27	16.16	0.03
solvent	lig_ejm_46	lig_jmc_27	16.01	0.03
solvent	lig_ejm_46	lig_jmc_27	16.07	0.03
complex	lig_ejm_46	lig_jmc_28	23.43	0.04
complex	lig_ejm_46	lig_jmc_28	23.29	0.04
complex	lig_ejm_46	lig_jmc_28	23.17	0.04
solvent	lig_ejm_46	lig_jmc_28	23.67	0.03
solvent	lig_ejm_46	lig_jmc_28	23.61	0.03
solvent	lig_ejm_46	lig_jmc_28	23.65	0.03
"""


@pytest.mark.xfail
@pytest.mark.parametrize('report', ["", "dg", "ddg"])
def test_gather(results_dir, report):
    expected = {
        "": _EXPECTED_DG,
        "dg": _EXPECTED_DG,
        "ddg": _EXPECTED_DDG,
        "dg-raw": _EXPECTED_DG_RAW,
    }[report]
    runner = CliRunner()

    if report:
        args = ["--report", report]
    else:
        args = []

    result = runner.invoke(gather, ['results'] + args + ['-o', '-'])

    assert result.exit_code == 0

    actual_lines = set(result.stdout_bytes.split(b'\n'))

    assert set(expected.split(b'\n')) == actual_lines


@pytest.mark.parametrize('include', ['complex', 'solvent', 'vacuum'])
def test_generate_bad_legs_error_message(include):
    expected = {
        'complex': ("appears to be an RBFE", "missing {'solvent'}"),
        'vacuum': ("appears to be an RHFE", "missing {'solvent'}"),
        'solvent': ("whether this is an RBFE or an RHFE",
                    "'complex'", "'solvent'"),
    }[include]
    set_vals = {include}
    ligpair = {'lig1', 'lig2'}
    msg = _generate_bad_legs_error_message(set_vals, ligpair)
    for string in expected:
        assert string in msg


@pytest.mark.xfail
def test_missing_leg_error(results_dir):
    file_to_remove = "easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json"
    (pathlib.Path("results") / file_to_remove).unlink()

    runner = CliRunner()
    result = runner.invoke(gather, ['results'] + ['-o', '-'])
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "Unable to determine" in str(result.exception)
    assert "'lig_ejm_31'" in str(result.exception)
    assert "'lig_ejm_42'" in str(result.exception)


@pytest.mark.xfail
def test_missing_leg_allow_partial(results_dir):
    file_to_remove = "easy_rbfe_lig_ejm_31_complex_lig_ejm_42_complex.json"
    (pathlib.Path("results") / file_to_remove).unlink()

    runner = CliRunner()
    result = runner.invoke(gather,
                           ['results'] + ['--allow-partial', '-o', '-'])
    assert result.exit_code == 0


RBFE_RESULTS = pooch.create(
    pooch.os_cache('openfe'),
    base_url="doi:10.6084/m9.figshare.25148945",
    registry={"results.tar.gz": "bf27e728935b31360f95188f41807558156861f6d89b8a47854502a499481da3"},
)


@pytest.fixture
def rbfe_results():
    # fetches rbfe results from online
    # untars into local directory and returns path to this
    d = RBFE_RESULTS.fetch('results.tar.gz', processor=pooch.Untar())

    return os.path.join(pooch.os_cache('openfe'), 'results.tar.gz.untar', 'results')


@pytest.mark.download
@pytest.mark.xfail
def test_rbfe_results(rbfe_results):
    runner = CliRunner()

    result = runner.invoke(gather, ['--report', 'raw', rbfe_results])

    assert result.exit_code == 0
    assert result.stdout_bytes == _EXPECTED_RAW
