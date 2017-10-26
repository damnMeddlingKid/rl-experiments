from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['gym>=0.9.4']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My trainer application package.'
)


4239150080257023

def printRec(pattern, index=0, previous_message="")
	if index >= len(pattern)
		print previous_message 
		return

	if pattern[index] == '?':
		for bit in range(2):
			printRec(pattern, index+1, previous_message + bit)
	else: 
		printRec(pattern, index+1, previous_message + pattern[index])



def printHandler(pattern):
	printRec(pattern)
