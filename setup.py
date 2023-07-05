from setuptools import find_packages, setup
from setuptools.command.install import install


class DownloadNLTK(install):

    def run(self):
        self.do_egg_install()
        import nltk
        nltk.download('punkt')


with open('README.md') as f:
    readme = f.read()


def do_setup():
    setup(
        name='opencompass',
        version='0.5.0',
        description='A comprehensive toolkit for large model evaluation',
        # url="",
        # author="",
        long_description=readme,
        long_description_content_type='text/markdown',
        cmdclass={'download_nltk': DownloadNLTK},
        setup_requires=['nltk==3.8'],
        python_requires='>=3.8.0',
        packages=find_packages(exclude=[
            'test*',
            'paper_test*',
        ]),
        keywords=['AI', 'NLP', 'in-context learning'],
        classifiers=[
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
        ])


if __name__ == '__main__':
    do_setup()
