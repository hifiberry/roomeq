from setuptools import setup, find_packages
import os

def get_version():
    changelog_path = os.path.join(os.path.dirname(__file__), 'debian/changelog')
    try:
        with open(changelog_path, 'r') as f:
            first_line = f.readline()
            version = first_line.split('(')[1].split(')')[0]
            return version
    except Exception:
        return "unknown"

setup(
    name="roomeq",
    version=get_version(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["fastapi", "pyalsaaudio"],
    entry_points={
        'console_scripts': [
            'roomeq-server = roomeq.roomeq_server:app',
        ],
    },
    zip_safe=False,
)
