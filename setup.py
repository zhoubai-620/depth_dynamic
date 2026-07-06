from setuptools import setup, find_packages

setup(
    name="extreme_avoid",
    version="1.0.0",
    description="Extreme Environment Dynamic Obstacle Tracking & Avoidance System",
    author="",
    packages=find_packages(include=["extreme_avoid", "extreme_avoid.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)
