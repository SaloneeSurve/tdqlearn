from setuptools import setup, find_packages

setup(
    name="tdqlearn",
    version="0.1.0",
    author="Your Name",
    author_email="your_email@example.com",
    description="A simple DQN implementation for TD learning in RL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tdqlearn",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "gymnasium"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
