import sys
import os

with open('training_upscale.txt') as f:
    for i in range(60):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_upscale/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_upscale/{}.npy'.format(fname, fname))

with open('testing_upscale.txt') as f:
    for i in range(240):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_upscale_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_upscale_testing/{}.npy'.format(fname, fname))

with open('training_downscale.txt') as f:
    for i in range(80):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_downscale/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_downscale/{}.npy'.format(fname, fname))

with open('testing_downscale.txt') as f:
    for i in range(320):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_downscale_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_downscale_testing/{}.npy'.format(fname, fname))

with open('training_maximal.txt') as f:
    for i in range(161):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_maximal/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_maximal/{}.npy'.format(fname, fname))

with open('testing_maximal.txt') as f:
    for i in range(644):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_maximal_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_maximal_testing/{}.npy'.format(fname, fname))

with open('training_minimal.txt') as f:
    for i in range(10):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_minimal/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_minimal/{}.npy'.format(fname, fname))

with open('testing_minimal.txt') as f:
    for i in range(790):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_minimal_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_minimal_testing/{}.npy'.format(fname, fname))
