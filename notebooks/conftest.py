import pytest

# output directory for data
@pytest.fixture(scope="session")
def out():
    return "data/thc"