import sys
import time

sys.path.insert(0, '/home/avisingh/Downloads/libs/libsvm-3.20/python/')

from svmutil import *

train_data_dir = '/home/avisingh/Datasets/kernels_hw2/leukemia/leu'
test_data_dir = '/home/avisingh/Datasets/kernels_hw2/leukemia/leu.t'
k = 5 #cross-validation fold

y_train, x_train = svm_read_problem(train_data_dir)
n = len(y_train)/k
feature_dim = 7129 #change it for your dataset
C = [0.01, 0.1, 1, 10, 100]
Gammas_raw = [0.01, 0.1, 1.0, 10, 100]
Gammas = [x/float(feature_dim) for x in Gammas_raw]
coefs = [0, 1]
degrees = [2, 3]
#C = [1,2]
overall_accuracy = []
hyperparams = []
start = time.time()
for coef in coefs:
	for degree in degrees:
		for gamma in Gammas:
			for c in C:
				parameters = '-s 0 -t 1 -c ' + str(c) + ' -g ' + str(gamma) + ' -d ' + str(degree) + ' -r ' + str(coef) + ' -q'
				accuracy = []
				for i in xrange(0, k):
					y_valid = y_train[i*n:(i+1)*n]
					y_learn = y_train[:i*n] + y_train[(i+1)*n:]
					x_valid = x_train[i*n:(i+1)*n]
					x_learn = x_train[:i*n] + x_train[(i+1)*n:]
					model = svm_train(y_learn, x_learn, parameters)
					p_label, p_acc, p_val = svm_predict(y_valid, x_valid, model, '-q')
					accuracy.append(p_acc[0])
				overall_accuracy.append(sum(accuracy)/len(accuracy))
				hyperparams.append((gamma, coef, degree, c))
				print '%f, %f, %f, %f, %f' % (degree, coef, gamma, c, overall_accuracy[-1])
		

end = time.time()

print 'training/validation time (in seconds):',  end - start
print 'number of models trained:', k*len(hyperparams)
print overall_accuracy
print hyperparams

hyperparams_opt = hyperparams[overall_accuracy.index(max(overall_accuracy))]
print hyperparams_opt

start = time.time()
parameters = '-s 0 -t 1 -c ' + str(hyperparams_opt[3]) + ' -g ' + str(hyperparams_opt[0]) + ' -d ' + str(hyperparams_opt[2]) + ' -r ' + str(hyperparams_opt[1]) +  ' -q'
y_test, x_test = svm_read_problem(test_data_dir)
model = svm_train(y_train, x_train, parameters)

start = time.time()
p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
end = time.time()
print 'test time (in seconds):',  end - start

print p_acc
