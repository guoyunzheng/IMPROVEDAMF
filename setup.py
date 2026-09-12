from setuptools import find_packages, setup

setup(
    name="adaptive-mean-flow",
    version="0.1.0",
    description="Adaptive Mean Flow and transformer policy baselines",
    packages=find_packages(),
    package_data={"source": ["config/*.yaml", "config/task/*/*.yaml", "config/vision_encoder/*.yaml"]},
    python_requires=">=3.10",
)
