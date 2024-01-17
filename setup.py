from setuptools import setup, find_packages

setup(
    name='dex-mani-mujoco',
    version='0.1',  # Update with your version number
    packages=find_packages(),
    install_requires=[
        'wheel',
        'gym',  # Add other dependencies as needed
        'mujoco_py',
        'mujoco',
        'pandas',
        'numpy',
        'roboticstoolbox-python',
        'rospy'
        # Add more dependencies here
    ],
)
