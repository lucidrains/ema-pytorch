from setuptools import setup, find_packages

setup(
  name = 'ema-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.4',
  license='MIT',
  description = 'Easy way to keep track of exponential moving average version of your pytorch module',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/ema-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'exponential moving average'
  ],
  install_requires=[
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
