from setuptools import setup
import os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + "/requirements.txt"
install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="ship_performance_service_interface",
    version="1.1.3",
    packages=["ship_performance_service_interface_lib"],
    install_requires=install_requires,
    url="",
    license="",
    author="jni",
    author_email="jorgen.b.nielsen@sintef.no",
    description="protobuf interface for ship-performance-service",
)
