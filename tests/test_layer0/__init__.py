"""
tests/test_layer0/conftest.py
==============================
Local conftest for all Layer 0 tests.

This conftest file is intentionally kept minimal.  It prevents pytest
from climbing up to the root tests/conftest.py (which imports main.py
and requires speciesnet / full model deps) when only running Layer 0 tests:

    pytest tests/test_layer0/ -v

pytest-asyncio mode is set to "auto" so all async test functions
are treated as coroutines automatically.
"""

import pytest


# ── pytest-asyncio configuration ──────────────────────────────────────────────
# Configuring asyncio_mode here avoids the need to decorate every async test
# with @pytest.mark.asyncio (though explicit marks still work fine).
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async (handled by pytest-asyncio)",
    )
