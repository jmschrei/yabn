# yagn.py
# Contact: Jacob Schreiber
#          jmschreiber91@gmail.com

"""
For detailed documentation and examples, see the README.
"""

cimport cython
from cython.view cimport array as cvarray
from libc.math cimport log as clog, sqrt as csqrt, exp as cexp
import math, random, itertools as it, sys, bisect
import networkx
import scipy.stats, scipy.sparse, scipy.special

if sys.version_info[0] > 2:
	# Set up for Python 3
	from functools import reduce
	xrange = range
	izip = zip
else:
	izip = it.izip

import numpy
cimport numpy

from matplotlib import pyplot

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2_PI = 2.50662827463

# Useful speed optimized functions
cdef inline double _log ( double x ):
	'''
	A wrapper for the c log function, by returning negative input if the
	input is 0.
	'''

	return clog( x ) if x > 0 else NEGINF

cdef inline int pair_int_max( int x, int y ):
	'''
	Calculate the maximum of a pair of two integers. This is
	significantly faster than the Python function max().
	'''

	return x if x > y else y

cdef inline double pair_lse( double x, double y ):
	'''
	Perform log-sum-exp on a pair of numbers in log space..  This is calculated
	as z = log( e**x + e**y ). However, this causes underflow sometimes
	when x or y are too negative. A simplification of this is thus
	z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
	the inputs are infinity, return infinity, and if either of the inputs
	are negative infinity, then simply return the other input.
	'''

	if x == INF or y == INF:
		return INF
	if x == NEGINF:
		return y
	if y == NEGINF:
		return x
	if x > y:
		return x + clog( cexp( y-x ) + 1 )
	return y + clog( cexp( x-y ) + 1 )

# Useful python-based array-intended operations
def log(value):
	"""
	Return the natural log of the given value, or - infinity if the value is 0.
	Can handle both scalar floats and numpy arrays.
	"""

	if isinstance( value, numpy.ndarray ):
		to_return = numpy.zeros(( value.shape ))
		to_return[ value > 0 ] = numpy.log( value[ value > 0 ] )
		to_return[ value == 0 ] = NEGINF
		return to_return
	return _log( value )
		
def exp(value):
	"""
	Return e^value, or 0 if the value is - infinity.
	"""
	
	return numpy.exp(value)

cdef class NormalDistribution( object ):
	"""
	A normal distribution based on a mean and standard deviation.
	"""

	cdef list parameters, summaries
	cdef str name

	def __init__( self, mean, std ):
		"""
		Make a new Normal distribution with the given mean mean and standard 
		deviation std.
		"""
		
		# Store the parameters
		self.parameters = [mean, std]
		self.summaries = []
		self.name = "NormalDistribution"

	def log_probability( self, symbol, epsilon=1E-4 ):
		"""
		What's the probability of the given float under this distribution?
		
		For distributions with 0 std, epsilon is the distance within which to 
		consider things equal to the mean.
		"""

		return self._log_probability( symbol, epsilon )

	cdef double _log_probability( self, double symbol, double epsilon ):
		"""
		Do the actual math here.
		"""

		cdef double mu = self.parameters[0], sigma = self.parameters[1]
		if sigma == 0.0:
			if abs( symbol - mu ) < epsilon:
				return 0
			else:
				return NEGINF
  
		return _log( 1.0 / ( sigma * SQRT_2_PI ) ) - ((symbol - mu) ** 2) /\
			(2 * sigma ** 2)

	def sample( self ):
		"""
		Sample from this normal distribution and return the value sampled.
		"""
		
		# This uses the same parameterization
		return random.normalvariate(*self.parameters)
		
	def from_sample( self, items, weights=None, inertia=0.0, min_std=0.01 ):
		"""
		Set the parameters of this Distribution to maximize the likelihood of 
		the given sample. Items holds some sort of sequence. If weights is 
		specified, it holds a sequence of value to weight each item by.
		
		min_std specifieds a lower limit on the learned standard deviation.
		"""

		# If the distribution is frozen, don't bother with any calculation
		if len(items) == 0 or self.frozen == True:
			# No sample, so just ignore it and keep our old parameters.
			return

		# Make it be a numpy array
		items = numpy.asarray(items)
		
		if weights is None:
			# Weight everything 1 if no weights specified
			weights = numpy.ones_like(items)
		else:
			# Force whatever we have to be a Numpy array
			weights = numpy.asarray(weights)
		
		if weights.sum() == 0:
			# Since negative weights are banned, we must have no data.
			# Don't change the parameters at all.
			return
		# The ML uniform distribution is just sample mean and sample std.
		# But we have to weight them. average does weighted mean for us, but 
		# weighted std requires a trick from Stack Overflow.
		# http://stackoverflow.com/a/2415343/402891
		# Take the mean
		mean = numpy.average(items, weights=weights)

		if len(weights[weights != 0]) > 1:
			# We want to do the std too, but only if more than one thing has a 
			# nonzero weight
			# First find the variance
			variance = (numpy.dot(items ** 2 - mean ** 2, weights) / 
				weights.sum())
				
			if variance >= 0:
				std = csqrt(variance)
			else:
				# May have a small negative variance on accident. Ignore and set
				# to 0.
				std = 0
		else:
			# Only one data point, can't update std
			std = self.parameters[1]    
		
		# Enforce min std
		std = max( numpy.array([std, min_std]) )
		
		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		prior_mean, prior_std = self.parameters
		self.parameters = [ prior_mean*inertia + mean*(1-inertia), 
							prior_std*inertia + std*(1-inertia) ]

	def summarize( self, items, weights=None ):
		'''
		Take in a series of items and their weights and reduce it down to a
		summary statistic to be used in training later.
		'''

		items = numpy.asarray( items )

		# Calculate weights. If none are provided, give uniform weights
		if weights is None:
			weights = numpy.ones_like( items )
		else:
			weights = numpy.asarray( weights )

		# Save the mean and variance, the summary statistics for a normal
		# distribution.
		mean = numpy.average( items, weights=weights )
		variance = numpy.dot( items**2 - mean**2, weights ) / weights.sum()

		# Append the mean, variance, and sum of the weights to give the weights
		# of these statistics.
		self.summaries.append( [ mean, variance, weights.sum() ] )
		

	def from_summaries( self, inertia=0.0 ):
		'''
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		'''

		# If no summaries stored or the summary is frozen, don't do anything.
		if len( self.summaries ) == 0 or self.frozen == True:
			return

		summaries = numpy.asarray( self.summaries )

		# Calculate the new mean and variance. 
		mean = numpy.average( summaries[:,0], weights=summaries[:,2] )
		variance = numpy.sum( [(v+m**2)*w for m, v, w in summaries] ) \
			/ summaries[:,2].sum() - mean**2

		if variance >= 0:
			std = csqrt(variance)
		else:
			std = 0

		# Get the previous parameters.
		prior_mean, prior_std = self.parameters

		# Calculate the new parameters, respecting inertia, with an inertia
		# of 0 being completely replacing the parameters, and an inertia of
		# 1 being to ignore new training data.
		self.parameters = [ prior_mean*inertia + mean*(1-inertia),
							prior_std*inertia + std*(1-inertia) ]
		self.summaries = []

cdef class Node( object ):
	'''
	A node, which takes in an emission distribution and a name.
	'''

	cdef public NormalDistribution distribution
	cdef public str name

	def __init__( self, distribution, name ):
		self.distribution = distribution
		self.name = name

cdef class Network( object ):
	"""
	Represents a Hidden Markov Model.
	"""
	cdef public str name
	cdef public object graph
	cdef public list nodes
	cdef int [:] in_edge_count, in_transitions, out_edge_count, out_transitions
	cdef double [:] in_transition_weights
	cdef double [:] out_transition_weights

	def __init__( self, name=None ):
		"""
		Make a new Hidden Markov Model. Name is an optional string used to name
		the model when output. Name may not contain spaces or newlines.
		
		If start and end are specified, they are used as start and end states 
		and new start and end states are not generated.
		"""
		
		# Save the name or make up a name.
		self.name = name or str( id(self) )

		# This holds a directed graph between states. Nodes in that graph are
		# State objects, so they're guaranteed never to conflict when composing
		# two distinct models
		self.graph = networkx.DiGraph()
		
	
	def __str__(self):
		"""
		Represent this HMM with it's name and states.
		"""
		
		return "{}:\n\t{}".format(self.name, "\n\t".join(map(str, self.nodes)))

	def state_count( self ):
		"""
		Returns the number of states present in the model.
		"""

		return len( self.nodes )

	def edge_count( self ):
		"""
		Returns the number of edges present in the model.
		"""

		return len( self.out_transition_log_weights )

	def dense_weight_matrix( self ):
		"""
		Returns the dense weight matrix. Useful if the edges of
		somewhat small models need to be analyzed.
		"""

		m = len(self.nodes)
		transition_weights = numpy.zeros( (m, m) )

		for i in xrange(m):
			for n in xrange( self.out_edge_count[i], self.out_edge_count[i+1] ):
				transition_weights[i, self.out_transitions[n]] = \
					self.out_transition_weights[n]

		return transition_weights


	def add_node( self, node ):
		"""
		Adds the given State to the model. It must not already be in the model,
		nor may it be part of any other model that will eventually be combined
		with this one.
		"""
		
		# Put it in the graph
		self.graph.add_node( node )

	def add_nodes( self, nodes ):
		"""
		Adds multiple states to the model at the same time. Basically just a
		helper function for the add_state method.
		"""

		for node in nodes:
			self.add_node( node )
		
	def add_edge( self, a, b, weight ):
		"""
		Add a transition from state a to state b with the given (non-log)
		probability. Both states must be in the HMM already. self.start and
		self.end are valid arguments here. Probabilities will be normalized
		such that every node has edges summing to 1. leaving that node, but
		only when the model is baked. 
		"""


		# Add the transition
		self.graph.add_edge(a, b, weight=weight )

	def add_edges( self, a, b, weights ):
		"""
		Add many transitions at the same time, in one of two forms. 

		(1) If both a and b are lists, then create transitions from the i-th 
		element of a to the i-th element of b with a probability equal to the
		i-th element of probabilities.

		Example: 
		model.add_transitions([model.start, s1], [s1, model.end], [1., 1.])

		(2) If either a or b are a state, and the other is a list, create a
		transition from all states in the list to the single state object with
		probabilities and pseudocounts specified appropriately.

		Example:
		model.add_transitions([model.start, s1, s2, s3], s4, [0.2, 0.4, 0.3, 0.9])
		model.add_transitions(model.start, [s1, s2, s3], [0.6, 0.2, 0.05])

		If a single group is given, it's assumed all edges should belong to that
		group. Otherwise, either groups can be a list of group identities, or
		simply None if no group is meant.
		"""

		n = len(a) if isinstance( a, list ) else len(b)


		# Allow addition of many transitions from many states
		if isinstance( a, list ) and isinstance( b, list ):
			# Set up an iterator across all edges
			for start, end, weight in izip( a, b, weights ):
				self.add_transition( start, end, weight )

		# Allow for multiple transitions to a specific state 
		elif isinstance( a, list ) and isinstance( b, Node ):
			# Set up an iterator across all edges to b
			for start, weight in izip( a, weights ):
				self.add_transition( start, b, weight )

		# Allow for multiple transitions from a specific state
		elif isinstance( a, Node ) and isinstance( b, list ):
			# Set up an iterator across all edges from a
			for end, weight in izip( b, weights ):
				self.add_transition( a, end, weight )


	def bake( self ): 
		"""
		Finalize the topology of the model, and assign a numerical index to
		every node. This method must be called before any of the probability-
		calculating or sampling methods.
		
		This fills in self.states (a list of all states in order) and 
		self.transition_log_probabilities (log probabilities for transitions).

		The option verbose will return a log of the changes made to the model
		due to normalization or merging. 
		"""

		# Go through the model and delete any nodes which have no edges leading
		# to it, or edges leading out of it. This gets rid of any states with
		# no edges in or out, as well as recursively removing any chains which
		# are impossible for the viterbi path to touch.
		self.in_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 ) 
		self.out_edge_count = numpy.zeros( len( self.graph.nodes() ), 
			dtype=numpy.int32 )


		self.nodes = self.graph.nodes()
		n, m = len(self.nodes), len(self.graph.edges())

		# We need a good way to get transition probabilities by state index that
		# isn't N^2 to build or store. So we will need a reverse of the above
		# mapping. It's awkward but asymptotically fine.
		indices = { self.nodes[i]: i for i in xrange(n) }

		# This holds numpy array indexed [a, b] to transition log probabilities 
		# from a to b, where a and b are state indices. It starts out saying all
		# transitions are impossible.
		self.in_transitions = numpy.zeros( m, dtype=numpy.int32 ) - 1
		self.in_edge_count = numpy.zeros( n+1, dtype=numpy.int32 ) 
		self.out_transitions = numpy.zeros( m, dtype=numpy.int32 ) - 1
		self.out_edge_count = numpy.zeros( n+1, dtype=numpy.int32 )
		self.in_transition_weights = numpy.zeros( m )
		self.out_transition_weights = numpy.zeros( m )

		# Now we need to find a way of storing in-edges for a state in a manner
		# that can be called in the cythonized methods below. This is basically
		# an inversion of the graph. We will do this by having two lists, one
		# list size number of nodes + 1, and one list size number of edges.
		# The node size list will store the beginning and end values in the
		# edge list that point to that node. The edge list will be ordered in
		# such a manner that all edges pointing to the same node are grouped
		# together. This will allow us to run the algorithms in time
		# nodes*edges instead of nodes*nodes.

		for a, b in self.graph.edges_iter():
			# Increment the total number of edges going to node b.
			self.in_edge_count[ indices[b]+1 ] += 1
			# Increment the total number of edges leaving node a.
			self.out_edge_count[ indices[a]+1 ] += 1

		# Take the cumulative sum so that we can associate array indices with
		# in or out transitions
		self.in_edge_count = numpy.cumsum(self.in_edge_count, 
			dtype=numpy.int32)
		self.out_edge_count = numpy.cumsum(self.out_edge_count, 
			dtype=numpy.int32 )

		# Now we go through the edges again in order to both fill in the
		# transition probability matrix, and also to store the indices sorted
		# by the end-node.
		for a, b, data in self.graph.edges_iter( data=True ):
			# Put the edge in the dict. Its weight is log-probability
			start = self.in_edge_count[ indices[b] ]

			# Start at the beginning of the section marked off for node b.
			# If another node is already there, keep walking down the list
			# until you find a -1 meaning a node hasn't been put there yet.
			while self.in_transitions[ start ] != -1:
				if start == self.in_edge_count[ indices[b]+1 ]:
					break
				start += 1

			self.in_transition_weights[ start ] = data['weight']

			# Store transition info in an array where the in_edge_count shows
			# the mapping stuff.
			self.in_transitions[ start ] = indices[a]

			# Now do the same for out edges
			start = self.out_edge_count[ indices[a] ]

			while self.out_transitions[ start ] != -1:
				if start == self.out_edge_count[ indices[a]+1 ]:
					break
				start += 1

			self.out_transition_weights[ start ] = data['weight']
			self.out_transitions[ start ] = indices[b]  

	def sample( self ):
		'''
		Generate a random sample from this model.
		'''

		n = len( self.nodes )
		samples = [-1]*n
		visited = 0

		while True:
			for i in xrange( n ):
				if samples[i] != -1:
					continue

				sample = 0

				for k in xrange( self.in_edge_count[i], self.in_edge_count[i+1] ):
					ki = self.in_transitions[k]

					if samples[ki] == -1:
						sample = -1
						break
					else:
						sample += samples[ki]*self.in_transition_weights[k]  

				if sample >= 0:
					samples[i] = sample + self.nodes[i].distribution.sample()
					visited += 1

			if visited == n:
				break

		return samples