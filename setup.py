from setuptools import setup


setup(
    name="manipulatesound",
    version="0.0.1",
    description="Manipulation tasks with sound based on TDW.",
    python_requires=">=3.6",
    install_requires=["ikpy", "sounddevice", "soundfile", "numpy>=1.20.0", "dm_control", "tdw"],
)
