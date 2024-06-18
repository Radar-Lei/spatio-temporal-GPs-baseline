import urllib.request
import os


# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)

if not smoke_test and not os.path.isfile('datasets/elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', 'datasets/elevators.mat')