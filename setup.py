# In pyspeed_project/setup.py

from setuptools import setup, find_packages

setup(
    name="pyspeed",
    version="0.3.1",
    packages=find_packages(),
    author="AI Assistant",
    description="A prototype tool to profile and apply heuristic optimizations to Python code.",
    install_requires=[
        "customtkinter",
        "astor",
        "ipython",
        "requests", # For telemetry
        # Numba and NumPy are optional but recommended
    ],
    python_requires='>=3.8',
)