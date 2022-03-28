import setuptools

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name='happier',
    version='1.0.0',
    packages=[''],
    url='https://github.com/elias-ramzi/HAPPIER',
    license='MIT',
    author='Elias Ramzi',
    author_email='elias.ramzi@lecnam.net',
    description='Official code for the ECCV 2022 paper: Hierarchical Average Precision Training for Pertinent Image Retrieval.',
    python_requires='>=3.6',
    install_requires=install_requires
)
