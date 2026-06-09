import pytest


@pytest.fixture(scope="session")
def synthetic_data_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data")
