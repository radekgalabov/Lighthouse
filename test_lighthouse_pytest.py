import numpy as np
from io import StringIO
import pytest
import lighthouse_functions as lf

alfa_true = 5
beta_true = 0.5
alfa = np.array(np.linspace(-10, 10, 201))
beta = np.array(np.linspace(0.01, 5, 500))


def test_input_prior_parameters(monkeypatch):
    prior_test_inputs = StringIO('-11\n-2\n20\n-4\n30\n')
    monkeypatch.setattr('sys.stdin', prior_test_inputs)
    with pytest.raises(Exception):
        lf.input_prior_parameters()


def test_input_number_of_samples(monkeypatch):
    prior_test_input = StringIO('1\n1\n1\n1\n1.3\n')
    monkeypatch.setattr('sys.stdin', prior_test_input)
    with pytest.raises(Exception):
        lf.input_prior_parameters()


def test_compute_prior():
    compute_prior_test = (5, 2, 0.7, 1)
    prior = lf.compute_prior(*compute_prior_test)
    sum_of_distribution = np.sum(prior)
    assert np.isclose(sum_of_distribution, 1)


def test_bounds_and_min(monkeypatch):
    prior_test_input = StringIO('1\n1\n1\n1\n30\n')
    monkeypatch.setattr('sys.stdin', prior_test_input)
    lkhd = lf.post_likelihood()
    estimates = lf.report(lkhd)
    assert estimates[0] <= estimates[1] <= estimates[2]
    assert estimates[3] <= estimates[4] <= estimates[5]
