from setuptools import setup, find_packages

setup(
    name="tensorfn",
    version="0.1.5",
    description="Non-opionated utility library for PyTorch",
    url="https://github.com/rosinality/tensorfn",
    author="Kim Seonghyeon",
    author_email="kim.seonghyeon@navercorp.com",
    license="MIT",
    install_requires=["torch>=1.1", "pydantic>=1.5", "pyhocon>=0.3.54"],
    packages=find_packages(),
)
