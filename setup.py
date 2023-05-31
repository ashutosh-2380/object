from setuptools import setup, find_packages

setup(
    name="streamlit-object-detection",
    version="1.0",
    author="Ashutosh Sharma",
    author_email="2002248",
    description="Object Detection with YOLOS using Streamlit",
    packages=find_packages(),
    install_requires=[
        "streamlit>=0.85.1",
        "torch",
        "transformers",
        "Pillow",
        "requests"
    ],
)
