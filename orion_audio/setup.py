from setuptools import find_packages, setup

package_name = 'orion_audio'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='DanielFLopez1620',
    maintainer_email='dfelipe.lopez@gmail.com',
    description='Microphone capture and Voice Activity Detection for orion_interaction',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'audio_capture = orion_audio.audio_capture_node:main',
        ],
    },
)
