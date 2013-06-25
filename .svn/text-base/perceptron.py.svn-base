import numpy as np
import math
import matplotlib.pyplot as plt

INPUT_SIZE = 784
NUM_EXAMPLES = 2000

THREE = 1
FIVE = -1 


class Perceptron(object):
	'''A Parent Perceptron Object'''
	def __init__(self, train):
		''' train is the set of input vectors (X's), and Y is the
		prediction vector (Y_hat)
		'''
		self.train = train
		self.Y = np.zeros(len(self.train))

class LinearPerceptron(Perceptron):
	'''A linear perceptron'''
	def __init__(self, train):
		'''self.w is the vector of coefficients '''
		super(LinearPerceptron, self).__init__(train)
		self.w = np.zeros(INPUT_SIZE)

	def reset(self):
		'''Resets the coefficients to 0'''
		self.w = np.zeros(INPUT_SIZE)

	def online(self, labels, num_examples=NUM_EXAMPLES):
		'''Online perceptron algorithm'''
		for t in range(num_examples):
			if np.inner(self.w, self.train[t]) >= 0:
				self.Y[t] = THREE
			else:
				self.Y[t] = FIVE

			if self.Y[t] == FIVE and labels[t] == THREE:
				self.w += self.train[t]
			elif self.Y[t] == THREE and labels[t] == FIVE:
				self.w -= self.train[t]

		# Now, check how well we did!
		num_wrong = 0.0
		for i in range(num_examples):
			if self.Y[i] != labels[i]:
				num_wrong += 1

		# print "Linear Perceptron -- Online"
		# print "Number Of Examples: ", num_examples
		# print "Number Right: ", num_examples - num_wrong
		# print "Number Wrong: ", num_wrong
		# print "Error Rate: ", num_wrong / num_examples
		return num_wrong

	def batch_mode(self, labels, num_rounds, examples_per_round=NUM_EXAMPLES):
		'''Batch perceptron algorithm'''
		self.reset() # Just so we know we're starting from scratch
		for _ in range(num_rounds):
			self.online(labels, examples_per_round)

	def testing_error(self, test, test_labels):
		'''Calculates the testing error (number mistakes / total)'''
		predictions = np.zeros(len(test))
		for t in range(len(test)):
			if np.inner(self.w, test[t]) >= 0:
				predictions[t] = THREE
			else:
				predictions[t] = FIVE

		num_wrong = 0.0
		for i in range(len(predictions)):
			if predictions[i] != test_labels[i]:
				num_wrong += 1

		return num_wrong / len(test)



	def cross_validation_error(self, labels, num_rounds):
		'''Split up the input into 5 equal sizes. Use 4 of the pieces as 
		a training set, and 1 as a test set, and get the error. Returns the 
		average of these five errors.
		'''
		train1 = self.train[:400]
		train2 = self.train[400:800]
		train3 = self.train[800:1200]
		train4 = self.train[1200:1600]
		train5 = self.train[1600:]

		label1 = labels[:400]
		label2 = labels[400:800]
		label3 = labels[800:1200]
		label4 = labels[1200:1600]
		label5 = labels[1600:]

		all_sets = [train1, train2, train3, train4, train5]
		all_labels = [label1, label2, label3, label4, label5]

		errors = []
		for i in range(5):
			# List comprehension FTW!
			training_set = [digit for j, train in enumerate(all_sets) if j != i for digit in train]
			train_labels = [l for j, label in enumerate(all_labels) if j != i for l in label]
			
			testing_set = all_sets[i]
			test_labels = all_labels[i]

			p = LinearPerceptron(training_set)
			#print train_labels
			p.batch_mode(train_labels, num_rounds, len(train_labels))
			error = p.testing_error(testing_set, test_labels)
			errors.append(error)

		return np.mean(errors)

	def predict(self, labels):
		''' Generates the test200.label.linear.35 file'''
		num_rounds = 3
		self.reset()
		self.batch_mode(labels, num_rounds)

		testing_file = open("data/test200.databw.35", 'r')
		test = np.genfromtxt(testing_file)

		predictions = np.zeros(len(test))
		for i in range(len(test)):
			if np.inner(self.w, test[i]) >= 0:
				predictions[i] = THREE
			else:
				predictions[i] = FIVE

		predict_file = open("data/test200.label.linear.35", 'w')
		for p in predictions:
			predict_file.write(str(int(p)) + '\n')


class GaussianPerceptron(Perceptron):
	'''Gaussian perceptron object'''
	def __init__(self, X, sigma):
		'''self.c is the vector of coefficients, and
		self.sigma is the sigma used in the gaussian kernel function.
		'''
		super(GaussianPerceptron, self).__init__(X)
		self.c = np.zeros(NUM_EXAMPLES)
		self.sigma = sigma

	def hyp(self, t):
		'''The hypothesis function'''
		s = 0
		for i in range(t):
			s += self.c[i] * gaussian_kernel(self.train[i], self.train[t], self.sigma)
		return s

	def hyp_all(self, t):
		'''The hypothesis function used for the test method (when we have already 
			initialized all the c values).
		'''
		s = 0
		for i in range(len(self.train)):
			s += self.c[i] * gaussian_kernel(self.train[i], self.train[t], self.sigma)
		return s

	def reset(self):
		'''Resets the coefficients to zero'''
		self.c = np.zeros(NUM_EXAMPLES)

	def online(self, labels, num_examples=NUM_EXAMPLES):
		'''Online algorithm'''
		for t in range(num_examples):
			if self.hyp(t) >= 0:
				self.Y[t] = THREE
			else:
				self.Y[t] = FIVE

			if self.Y[t] == FIVE and labels[t] == THREE:
				self.c[t] += 1
			elif self.Y[t] == THREE and labels[t] == FIVE:
				self.c[t] += -1

		# Now, check how well we did!
		num_wrong = 0.0
		for i in range(num_examples):
			if self.Y[i] != labels[i]:
				num_wrong += 1

		# print "Kernel (Gaussian) Perceptron -- Online"
		# print "Number Of Examples: ", num_examples
		# print "Number Right: ", num_examples - num_wrong
		# print "Number Wrong: ", num_wrong
		# print "Error Rate: ", num_wrong / num_examples
		return num_wrong

	def batch_mode(self, labels, num_rounds, examples_per_round=NUM_EXAMPLES):
		'''Batch mode algorithm'''
		self.reset() # Just so we know we're starting from scratch
		errors = []
		for i in range(num_rounds):
			print i
			num_wrong = self.online(labels, examples_per_round)
			errors.append(num_wrong)

	def testing_error(self, test, test_labels):
		'''Calculates the testing error.'''
		predictions = np.zeros(len(test))
		for t in range(len(test)):
			if self.hyp_all(t) >= 0:
				predictions[t] = THREE
			else:
				predictions[t] = FIVE

		num_wrong = 0.0
		for i in range(len(predictions)):
			if predictions[i] != test_labels[i]:
				num_wrong += 1

		return num_wrong / len(test)



	def cross_validation_error(self, labels, num_rounds, sigma):
		'''Split up the input into 5 equal sizes. Use 4 of the pieces as 
		a training set, and 1 as a test set, and get the error. Returns the 
		average of these five errors.
		'''
		train1 = self.train[:400]
		train2 = self.train[400:800]
		train3 = self.train[800:1200]
		train4 = self.train[1200:1600]
		train5 = self.train[1600:]

		label1 = labels[:400]
		label2 = labels[400:800]
		label3 = labels[800:1200]
		label4 = labels[1200:1600]
		label5 = labels[1600:]

		all_sets = [train1, train2, train3, train4, train5]
		all_labels = [label1, label2, label3, label4, label5]

		errors = []
		for i in range(5):
			# List comprehension FTW!
			training_set = [digit for j, train in enumerate(all_sets) if j != i for digit in train]
			train_labels = [l for j, label in enumerate(all_labels) if j != i for l in label]
			
			testing_set = all_sets[i]
			test_labels = all_labels[i]

			p = GaussianPerceptron(training_set, sigma)
			p.batch_mode(train_labels, num_rounds, len(train_labels))
			error = p.testing_error(testing_set, test_labels)
			errors.append(error)

		return np.mean(errors)

	def predict(self, labels):
		num_rounds = 3
		self.reset()
		self.batch_mode(labels, num_rounds)

		testing_file = open("data/test200.databw.35", 'r')
		test = np.genfromtxt(testing_file)

		predictions = np.zeros(len(test))
		for i in range(len(test)):
			# hyp = 0
			# for j in range(len(predictions)):
			# 	hyp += self.c[j] * gaussian_kernel(predictions[j], predictions[i], self.sigma)
			if self.hyp_all(i) >= 0:
				predictions[i] = THREE
			else:
				predictions[i] = FIVE

		predict_file = open("data/test200.label.kernel.35", 'w')
		for p in predictions:
			predict_file.write(str(int(p)) + '\n')

def gaussian_kernel(v1, v2, sigma):
	''' Computes the Gaussian kernel function '''
	norm = np.linalg.norm(v1 - v2)
	return math.exp(((- (norm**2.0)) / (2 * sigma**2)))

def plot1(train, labels):
	''' Plots the number of mistakes made by the linear perceptron as 
	a function of the number of examples.
	'''
	p = LinearPerceptron(train)
	example_nums_small = [20*(i+1) for i in range(9)] # Because the data is tighter here
	example_nums_large = [200*(i+1) for i in range(10)]
	example_nums = example_nums_small + example_nums_large
	mistakes = []
	for num in example_nums:
		print num
		mistake_count = p.online(labels, num)
		mistakes.append(mistake_count)
		p.reset()

	plt.plot(example_nums, mistakes)
	plt.xlabel("Number of examples seen")
	plt.ylabel("Number of mistakes made")
	plt.ylim([0, 300])
	plt.show()

def plot2(train, labels):
	''' Plots the number of mistakes made by the Gaussian perceptron as 
	a function of the number of examples.
	'''
	p = GaussianPerceptron(train, 5)
	example_nums_small = [20*(i+1) for i in range(9)] # Because the data is tighter here
	example_nums_large = [200*(i+1) for i in range(10)]
	example_nums = example_nums_small + example_nums_large
	mistakes = []
	for num in example_nums:
		print num
		mistake_count = p.online(labels, num)
		mistakes.append(mistake_count)
		p.reset()

	plt.plot(example_nums, mistakes)
	plt.xlabel("Number of examples seen")
	plt.ylabel("Number of mistakes made")
	plt.ylim([0, 300])
	plt.show()

def plot3(train, labels):
	'''Plots the cross validation error for multiple sigma values'''
	sigmas = [1, 3, 5, 7]
	mistakes = []
	for sig in sigmas:
		print sig
		p = GaussianPerceptron(train, sig)
		num_wrong = p.cross_validation_error(labels, 1, sig)
		mistakes.append(num_wrong)

	plt.plot(sigmas, mistakes)
	plt.xlabel("Sigma")
	plt.ylabel("Cross-validation Error")
	plt.xlim([0, 8])
	plt.ylim([0, .2])
	print mistakes
	plt.show()

def plot4(train, labels):
	'''Plots the cross validation error for multiple times run.'''
	num_rounds = list(range(20))
	cross_errors = []
	for round in num_rounds:
		p = LinearPerceptron(train)
		cross_errors.append(p.cross_validation_error(labels, round))

	plt.plot(num_rounds, cross_errors)
	plt.xlabel("Number of times the data was cycled through")
	plt.ylabel("Cross-Validation Error")
	plt.show()







if __name__ == "__main__":
	train_fp = "data/train2k.databw.35"
	label_fp = "data/train2k.label.35"
	train_file = open(train_fp, 'r')
	label_file = open(label_fp, 'r')
	sigma = 5

	train = np.genfromtxt(train_file)
	labels = np.genfromtxt(label_file)

	p = GaussianPerceptron(train, sigma)
	p.predict(labels)

	# p = LinearPerceptron(train)
	# print p.cross_validation_error(labels, 10)
	# p.online(labels)

	# p = GaussianPerceptron(train, 5)
	# p.online(labels)

	#plot3(train, labels)



