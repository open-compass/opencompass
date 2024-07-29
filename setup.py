from setuptools import find_packages, setup
from setuptools.command.install import install


class DownloadNLTK(install):

    def run(self):
        self.do_egg_install()
        import nltk
        nltk.download('punkt')


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    if '--' in version:
                        # the `extras_require` doesn't accept options.
                        version = version.split('--')[0].strip()
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def get_version():
    version_file = 'opencompass/__init__.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def do_setup():
    setup(
        name='opencompass',
        author='OpenCompass Contributors',
        version=get_version(),
        description='A comprehensive toolkit for large model evaluation',
        url='https://github.com/open-compass/opencompass',
        long_description=readme(),
        long_description_content_type='text/markdown',
        maintainer='OpenCompass Authors',
        cmdclass={'download_nltk': DownloadNLTK},
        setup_requires=['nltk==3.8'],
        python_requires='>=3.8.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        license='Apache License 2.0',
        include_package_data=True,
        packages=find_packages(),
        keywords=[
            'AI', 'NLP', 'in-context learning', 'large language model',
            'evaluation', 'benchmark', 'llm'
        ],
        classifiers=[
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
        ],
        entry_points={
            'console_scripts': [
                'opencompass = opencompass.cli.main:main',
            ],
        },
    )


if __name__ == '__main__':
    do_setup()
