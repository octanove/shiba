import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shiba-model",
    version="0.1.0",
    author="Octanove Labs",
    author_email="mindful.jt@gmail.com",
    description="An efficient character-level transformer encoder, pretrained for Japanese",
    keywords=['natural language processing', 'deep learning', 'transformer'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/octanove/shiba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["training", "training.*"]),
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "local-attention == 1.4.1",
    ]
)
