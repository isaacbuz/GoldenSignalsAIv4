from setuptools import setup, find_packages

setup(
    name="golden_signals_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
    ],
    python_requires=">=3.8",
) 