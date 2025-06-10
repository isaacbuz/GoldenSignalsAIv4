from setuptools import setup, find_packages

setup(
    name="AlphaPy",
    version="1.0.0",
    description="Algorithmic Trading Library for GoldenSignalsAI",
    author="GoldenSignalsAI Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0"
    ],
    extras_require={
        'dev': [
            'black>=21.5b2',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'pytest-mock>=3.6.0'
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment"
    ]
) 