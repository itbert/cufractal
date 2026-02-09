from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import sys
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build_args = ["--config", "Release"]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            build_args += ["--", "-j4"]
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )

setup(
    name="cufractal",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="GPU-accelerated fractal generator with CUDA",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cufractal",
    packages=["cufractal_py"],
    package_dir={"cufractal_py": "python"},
    ext_modules=[CMakeExtension("cufractal")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy>=1.19.0", "pybind11>=2.6.0"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
