
from setuptools import setup, find_packages

setup(
	name = "nxopt",
	description = "A pretty ugly network optimization library, more for convenience than anything",
	author = "Kyle Treleaven",
	author_email = "ktreleav@gmail.com",
	version = "0.0.0",
	packages = find_packages(),
	namespace_packages = [ 'setiptah', 'setiptah.nxopt', ],
)

