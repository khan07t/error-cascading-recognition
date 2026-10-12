from setuptools import find_packages, setup

setup(
    name="central-logging",
    version="0.1.0",
    description="Error cascade recognition for PLC alarm logs",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "hdbscan>=0.8.33",
        "torch>=2.1",
        "transformers>=4.40,<5",
    ],
    extras_require={"dashboard": ["dash>=2.14", "plotly>=5.18"],
                    "notebooks": ["jupyter", "matplotlib>=3.7"],
                    "dev": ["pytest>=7.4"]},
    entry_points={"console_scripts": ["central-logging=central_logging.main:cli"]},
)
