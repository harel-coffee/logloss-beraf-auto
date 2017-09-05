import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest  # import here, because outside the required eggs aren't loaded yet
        sys.exit(pytest.main(self.test_args))

setup(
    name='logloss-beraf',
    version="0.7",
    description="A tool for costructing a limited sized diagnostic panels based on methylation data",
    long_description=open("README.md").read(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python'
    ],
    keywords="model machine learning diagnostic panel methylation",
    author="dezzan",
    author_email="dezzandev@gmail.com",
    url="",
    packages=find_packages(exclude=('tests', 'tests.*')),
    license='GPLv3',
    install_requires=open('./requirements.txt').read(),
    include_package_data=True,
    zip_safe=False,
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    entry_points={
        'console_scripts':
            ['logloss_beraf=logloss_beraf:main']
      }

)
