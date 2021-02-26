import setuptools

setuptools.setup(
    name="graphtorch",
    version="0.1.1",
    license='MIT',
    author="Hyeonwoo Daniel Yoo",
    author_email="hyeon95y@gmail.com",
    description="From adjacency, feature matrix to pytorch module",
    long_description=open('README.md').read(),
    url="https://github.com/hyeon95y/graphtorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
