from setuptools import setup, find_packages

setup(
    name="kernelmind",
    version="0.1.1",
    description="A minimalist thinking kernel for LLM agents",
    author="Ethan Zhan",
    author_email="jiezhan1@gmail.com",
    url="https://github.com/zhanj/kernelmind-framework",
    packages=find_packages(),
    install_requires=["openai", "pyyaml", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
