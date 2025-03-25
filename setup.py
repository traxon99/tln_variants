from setuptools import setup

package_name = 'tln_variants'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jackson',
    maintainer_email='jackson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'tln_standard = tln_variants.tln_standard:main',
            'tln_temporal = tln_variants.tln_temporal:main',
            'evaluation = tln_variants.evaluation:main',
            'collect = tln_variants.data_collection:main'
        ],
    },
)
