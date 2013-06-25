
import numpy as np
import matplotlib.pyplot as plt
import math

INPUT_SIZE = 784
NUM_EXAMPLES = 2000
THREE = 1
FIVE = -1

# In this case, a '1' will signal a three, and a '0' will signal a 5
class Perceptron(object):
	def __init__(self, X):
		self.X = X
		self.Y = np.zeros(NUM_EXAMPLES)

	def check_accuracy(self, labels):
		num_right = 0
		for i in range(len(labels)):
			if self.Y[i] == 1 and labels[i] == THREE:
				num_right += 1
			elif self.Y[i] == 0 and labels[i] == FIVE:
				num_right += 1

		return num_right



class LinearPerceptron(Perceptron):
	def __init__(self, X, w=np.zeros(INPUT_SIZE)):
		super(LinearPerceptron, self).__init__(X)
		self.w = w

	def perceptron(self, labels, counter=NUM_EXAMPLES):
		for i in range(counter):
			t = i % NUM_EXAMPLES
			if np.inner(self.w, self.X[t]) >= 0:
				self.Y[t] = 1
			else:
				self.Y[t] = 0

			true_answer = labels[t]
			if self.Y[t] == 1 and labels[t] == FIVE:
				self.w -= self.X[t]
			elif self.Y[t] == 0 and labels[t] == THREE:
				self.w += self.X[t]

	def predict(self, test):
		test_len = len(test)
		answers = np.zeros(test_len)
		for t in range(test_len):
			if np.inner(self.w, test[t]) >= 0:
				answers[t] = THREE
			else:
				answers[t] = FIVE
		return answers


class KernelPerceptron(Perceptron):
	def __init__(self, X, kernel, c=np.zeros(NUM_EXAMPLES)):
		super(KernelPerceptron, self).__init__(X)
		self.c = c
		self.kernel = kernel

	def perceptron(self, labels, sigma, counter=NUM_EXAMPLES):
		for i in range(counter):
			t = i % NUM_EXAMPLES
			hyp = 0
			for j in range(counter):
				hyp += self.c[j]*self.kernel(self.X[j], self.X[t], sigma)
			if hyp >= 0:
				self.Y[t] = 1
			else:
				self.Y[t] = 0

			true_answer = labels[t]
			if self.Y[t] == 1 and labels[t] == FIVE:
				self.c[t] = -1
			elif self.Y[t] == 0 and labels[t] == THREE:
				self.c[t] = 1

	def predict(self, test, sigma):
		test_len = len(test)
		answers = np.zeros(test_len)
		for t in range(test_len):
			hyp = 0
			for j in range(test_len):
				hyp += self.c[j]*self.kernel(test[j], test[t], sigma)
				print self.kernel(test[j], test[t], sigma)
			if hyp >= 0:
				answers[t] = THREE
			else:
				answers[t] = FIVE
		return answers


def gaussian_kernel(v1, v2, sigma):
	norm = np.linalg.norm(v1 - v2)
	print norm
	return math.exp(((- (norm**2)) / (2 * sigma**2)))

def linear_kernel(v1, v2, sigma=None):
	return np.inner(v1, v2)


def online_mode(X, labels, kernel_mode, examples_seen=NUM_EXAMPLES):

	if kernel_mode:
		kernel = gaussian_kernel
		#kernel = linear_kernel
		p = KernelPerceptron(X, kernel)
		sigma = 5
		p.perceptron(labels, sigma, examples_seen)

		num_right = p.check_accuracy(labels[:examples_seen])
		return num_right

	else:
		p = LinearPerceptron(X)
		p.perceptron(labels, examples_seen)

		num_right = p.check_accuracy(labels[:examples_seen])
		return num_right

#def batch_mode(X, labels, kernel_mode):

def cross_validate(X, labels, kernel_mode):
	training = X[:1500]
	test = X[1501:]
	training_lab = labels[:1500]
	test_lab = labels[1501:]
	sigma = 5
	if kernel_mode:
		p = KernelPerceptron(training, gaussian_kernel)
		p.perceptron(training_lab, sigma, 1500)
		answers = p.predict(test, sigma)
		err = get_error(answers, test_lab)
	else:
		p = LinearPerceptron(training)
		p.perceptron(training_lab, 1500)
		answers = p.predict(test)
		err = get_error(answers, test_lab)
	return err

def get_error(answers, test_labels):
	num_wrong = 0.0
	for i in range(len(answers)):
		if answers[i] != test_labels[i]:
			num_wrong += 1
	return num_wrong / len(answers)







def plot_mistakes_linear(X, labels):
	example_nums = [25*(i+1) for i in range(80)]
	num_mistakes = []
	for num in example_nums:
		print num
		num_right = online_mode(X, labels, False, num)
		num_mistakes.append(num - num_right)

	plt.plot(example_nums, num_mistakes)
	plt.ylim([0, 100]) 
	# accuracy = [num_mistake / float(total) for (num_mistake, total) in zip(num_mistakes, example_nums)]
	# plt.plot(example_nums, accuracy)
	plt.xlabel("Number of examples seen")
	plt.ylabel("Number of mistakes made")
	plt.show()

def plot_mistakes_kernel(X, labels):
	example_nums_small = [20*(i+1) for i in range(9)]
	example_nums_large = [200*(i+1) for i in range(10)]
	example_nums = example_nums_small + example_nums_large
	num_mistakes = []
	for num in example_nums:
		print num
		num_right = online_mode(X, labels, True, num)
		num_mistakes.append(num - num_right)

	plt.plot(example_nums, num_mistakes)
	plt.ylim([0, 300])
	plt.xlabel("Number of examples seen")
	plt.ylabel("Number of mistakes made")
	plt.show()



if __name__ == "__main__":
	train_fp = "data/train2k.databw.35"
	label_fp = "data/train2k.label.35"
	train_file = open(train_fp, 'r')
	label_file = open(label_fp, 'r')

	X = np.genfromtxt(train_file)
	labels = np.genfromtxt(label_file)



	#err = cross_validate(X, labels, True)
	#print err
	p = KernelPerceptron(X, gaussian_kernel)
	p.perceptron(labels, 5)
	print p.predict(X, 5)

	#plot_mistakes_linear(X, labels)
	#plot_mistakes_kernel(X, labels)
	# num_examples = 2000
	# right = online_mode(train_file, label_file, True, num_examples)
	# print float(right) / min(num_examples, NUM_EXAMPLES)



