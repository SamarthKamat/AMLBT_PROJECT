import sys
import warnings
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
warnings.filterwarnings("ignore")

from context import ExecContext  # noqa: E402
from data_profiler import profile_dataframe  # noqa: E402
from tests import datasets  # noqa: E402
from validator import roles_from_profile  # noqa: E402


def make_ctx(df, name="test"):
    profile = profile_dataframe(df, name)
    roles = roles_from_profile(profile)
    return profile, roles, ExecContext(roles=roles, profile=profile, dataset_id=name)


@pytest.fixture(scope="session")
def delivery():
    df = datasets.delivery_df()
    return (df, *make_ctx(df, "delivery"))


@pytest.fixture(scope="session")
def hr():
    df = datasets.hr_df()
    return (df, *make_ctx(df, "hr"))


@pytest.fixture(scope="session")
def retail():
    df = datasets.retail_df()
    return (df, *make_ctx(df, "retail"))
