import sys
import time

sys.path.insert(0, '/home/avisingh/Downloads/libs/liblinear-2.1/python/')

from liblinearutil import *

train_data_dir = '/home/avisingh/Datasets/kernels_hw2/rcv1.binary/rcv1_train.binary'
test_data_dir = '/home/avisingh/Datasets/kernels_hw2/rcv1.binary/rcv1_test.binary'
k = 5 #cross-validation fold

y_train, x_train = svm_read_problem(train_data_dir)
n = len(y_train)/k

C = [0.01, 0.1, 1, 10, 100]
#C = [1,2]
overall_accuracy = []

start = time.time()
for c in C:
	parameters = '-s 2 -c ' + str(c) + ' -q'
	accuracy = []
	for i in xrange(0, k):
		y_valid = y_train[i*n:(i+1)*n]
		y_learn = y_train[:i*n] + y_train[(i+1)*n:]
		x_valid = x_train[i*n:(i+1)*n]
		x_learn = x_train[:i*n] + x_train[(i+1)*n:]
		model = train(y_learn, x_learn, parameters)
		p_label, p_acc, p_val = predict(y_valid, x_valid, model, '-q')
		accuracy.append(p_acc[0])
	overall_accuracy.append(sum(accuracy)/len(accuracy))
	print overall_accuracy[-1]

end = time.time()

print 'training/validation time (in seconds):',  end - start
print 'number of models trained:', k*len(C)
print overall_accuracy
print C

c_opt = C[overall_accuracy.index(max(overall_accuracy))]
print c_opt

start = time.time()
parameters = '-s 2 -c ' + str(c_opt) + ' -q'
y_test, x_test = svm_read_problem(test_data_dir)
model = train(y_train, x_train, parameters)

start = time.time()
p_label, p_acc, p_val = predict(y_test, x_test, model)
end = time.time()
print 'test time (in seconds):',  end - start

print p_acc
