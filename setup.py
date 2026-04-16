from setuptools import setup, Extension

# Define the C++ extension module
rocket_extension = Extension(
    'rocket_core',
    sources=['src/rocket.cpp'],
    include_dirs=['src'],
    language='c++'
)

setup(
    name='rocket_core',
    version='0.1.0',
    description='Core C++ engine for Rocket-Lib',
    ext_modules=[rocket_extension],
)
