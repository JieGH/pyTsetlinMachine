from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(2000, 50, 10.0)

print("\nAccuracy over 400 epochs:\n")
for i in range(400):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	tm_results = 100*(tm.predict(X_test) == Y_test).mean())
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, tm_results[-1], stop-start))