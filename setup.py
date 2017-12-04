from distutils.core import setup

from pip.req import parse_requirements

setup(
    name='nevermind',
    version='0.1',
    packages=['nevermind'],
    url='https://github.com/JuliusKunze/nevermind',
    license='MIT License',
    author='Julius Kunze',
    author_email='juliuskunze@gmail.com',
    description='',
    install_requires=[str(r.req) for r in parse_requirements('requirements.txt', session=False)]
)
