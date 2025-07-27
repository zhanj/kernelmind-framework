from setuptools import setup, find_packages

setup(
    name="kernelmind",
    version="0.1.0",
    description="A minimalist thinking kernel for LLM agents",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/your-org/kernelmind",
    packages=find_packages(),
    install_requires=["openai", "pyyaml", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
