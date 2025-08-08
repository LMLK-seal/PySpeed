# In pyspeed_project/pyspeed/__init__.py

def load_ipython_extension(ipython):
    """Load the magic extension."""
    from .magic import load_ipython_extension as load_magic
    load_magic(ipython)