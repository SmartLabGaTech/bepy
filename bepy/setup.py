from setuptools import setup

setup(name='bepy',
      version='0.209_test',
      description="A framework for materials characterization data collected that enables post analysis",
      url='https://github.com/lgriffin39/bepy',
      author='Lee Griffin',
      author_email='lgriffin39@gatech.edu',
      license='MIT',
      packages=['bepy'],
      install_requires=['numpy', 'pandas', 'matplotlib', 'scipy'],
      zip_safe=False)