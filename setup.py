from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kl-fixed-mixture",
    version="0.1.0",
    author="Michael Cohen",
    author_email="mkcohen@berkeley.edu",
    description="Outputs a probability distribution y between a and b with a desired KL(y || b); PyTorch-differentiable wrt a",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkc1000/kl-fixed-mixture",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.0.0',
    ],
)
