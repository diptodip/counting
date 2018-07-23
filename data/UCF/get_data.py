import sys
import os

with open('train_set_0.txt') as f:
    for i in range(40):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_0/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_0/{}.npy'.format(fname, fname))

with open('test_set_0.txt') as f:
    for i in range(10):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_0_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_0_testing/{}.npy'.format(fname, fname))

with open('train_set_1.txt') as f:
    for i in range(40):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_1/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_1/{}.npy'.format(fname, fname))

with open('test_set_1.txt') as f:
    for i in range(10):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_1_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_1_testing/{}.npy'.format(fname, fname))

with open('train_set_2.txt') as f:
    for i in range(40):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_2/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_2/{}.npy'.format(fname, fname))

with open('test_set_2.txt') as f:
    for i in range(10):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_2_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_2_testing/{}.npy'.format(fname, fname))

with open('train_set_3.txt') as f:
    for i in range(40):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_3/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_3/{}.npy'.format(fname, fname))

with open('test_set_3.txt') as f:
    for i in range(10):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_3_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_3_testing/{}.npy'.format(fname, fname))

with open('train_set_4.txt') as f:
    for i in range(40):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_4/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_4/{}.npy'.format(fname, fname))

with open('test_set_4.txt') as f:
    for i in range(10):
        fname = f.readline()
        fname = fname[:-1]
        os.system('cp A/{} A_4_testing/'.format(fname))
        fname = fname[:-4]
        os.system('cp B/{}dots.npy B_4_testing/{}.npy'.format(fname, fname))
