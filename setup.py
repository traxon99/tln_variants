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
            'ftg = tln_variants.ftg:main',
            'evaluation = tln_variants.evaluation:main',
            'collect = tln_variants.data_collection:main',
            'ftg_opp = tln_variants.ftg_opp:main',
            'joy_node_drive = tln_variants.joy:main',
            'joy_test = tln_variants.joy_test:main',
            'tln_override = tln_variants.tln_standard_override:main',
            'tln_vel = tln_variants.tln_standard_vel:main',
            'rln = tln_variants.rln:main',
            'rln_collect = tln_variants.rnn_data_collection:main',
            'rln_no_ts = tln_variants.rln_no_ts:main',
            'rln_sim = tln_variants.rln_sim:main'
        ],
    },
)
