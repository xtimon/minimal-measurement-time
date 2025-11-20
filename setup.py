"""
Setup script для установки пакета measurement-time-simulator
"""

from setuptools import setup, find_packages
import os

# Читаем README для длинного описания
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="minimal-measurement-time",
    version="0.1.0",
    author="Measurement Time Simulator Team",
    description="Фреймворк для симуляции времени измерения информации в физических системах",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/measurement-time-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda11x>=10.0.0; platform_machine=='x86_64'",
        ],
        "performance": [
            "numba>=0.56.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "all": [
            "cupy-cuda11x>=10.0.0; platform_machine=='x86_64'",
            "numba>=0.56.0",
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    keywords="physics simulation quantum measurement information theory",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/measurement-time-simulator/issues",
        "Source": "https://github.com/yourusername/measurement-time-simulator",
    },
)

