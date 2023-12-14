from setuptools import setup, find_packages

setup(
    name='shadow_hand',
    version='0.1',  # Update with your version number
    packages=find_packages(),
    install_requires=[
        'gym',  # Add other dependencies as needed
        'mujoco_py',
        # Add more dependencies here
    ],
)
