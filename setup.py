from setuptools import setup, find_packages

setup(
    name='dex-mani-mujoco',
    version='0.1',  # Update with your version number
    packages=find_packages(),
    install_requires=[
        'wheel',
        'gym',
        'mujoco_py',
        'mujoco',
        'pandas',
        'numpy',
        'roboticstoolbox-python',
        'rospy',
        # 'dm_control'
        # Add more dependencies here
    ],
)
