from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
import time

from keras.datasets import mnist

import argparse
parser = argparse.ArgumentParser(description='See readme for detail')
parser.add_argument("--c", help='Number of clauses in total	')
parser.add_argument("--t", help='Value of parameter T')
parser.add_argument("--s", help='Value of parameter S')
parser.add_argument("--e", help='Number of epochs for training')

args = parser.parse_args()
c = args.c
t = args.t
s = args.s
e = args.e

T = int(t)
s = float(s)
number_of_clauses = int(c)
epochs = int(e)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine( number_of_clauses, T, s)
timestr = time.strftime("%H%M%S%d%m%Y")

print ("Current run: clause=" ,number_of_clauses, "T=", T, "S=", s, 'time@',timestr)
epoch_range = np.arange(1,epochs+1,1)
for i in range(epochs):
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	result_test = 100*(tm.predict(X_test) == Y_test).mean()
	result_train = 100*(tm.predict(X_train) == Y_train).mean()
	print("# %d Epoch: Test accuracy: %.2f%%, Training accuracy: %.2f%%" % (i+1, result_test, result_train))
