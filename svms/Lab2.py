import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# GENERATING THE TEST DATA
numpy.random.seed(100)
classA = numpy.concatenate ( (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5], numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))   # Randoms from around 0.5 to 1.5 and -1.5 to 0.5
classB = numpy.random.randn(20, 2) * 0.2 + [0.0 , -0.5]        # Randoms from -0.5  to 0.0
inputs = numpy.concatenate ( (classA , classB) )    #x and y coordinates for the data
targets = numpy.concatenate (  ( numpy.ones( classA.shape[0]) , -numpy.ones( classB.shape [ 0 ] )))     # holds the +1 or -1 values for the inputs, their indices in the list coincide respectively
N = inputs.shape[0] # Number of rows (samples)

permute = list ( range(N) )
random.shuffle(permute)
inputs = inputs [ permute, : ]
targets = targets [ permute ]

print ( "Class A: " , classA )
print ( "Class B: " , classB )
print ( "N: " , N )
print ( "inputs: " , inputs )
print ( "targets: " , targets )




#                   HELPFUL TIP : LIST COMPREHENSIONS
# List Comprehensions: a way of creating a new list by looping over an already existing list or 'sequence'.
# this constructs a new list a of the same length as the sequence seq. Each
# element is computed by evaluating the expression expr while x temporarily contains
# the corresponding element from seq. Normally, expr is an
# expression which contains the variable x as a part.

# a = [ expr for x in seq ]








# This will find the vector ⃗α which minimizes the function objective within the
# bounds B and the constraints XC.

# objective is a function you have to define, which takes a vector ⃗α as argument and returns a scalar value
# it implements the expression that should be minimized (equation 4 in the pdf)

# start is a vector with the initial guess of the ⃗α vector. We can, e.g., simply
# use a vector of zeros: numpy.zeros(N). N is here the number of training samples
# (note that each training sample will have a corresponding α-value)

# B is a list of pairs of the same length as the ⃗α-vector, stating the lower and
# upper bounds for the corresponding element in ⃗α. To constrain the α values to
# be in the range 0 ≤ α ≤ C, we can set bounds=[(0, C) for b in range(N)].
# To only have a lower bound, set the upper bound to None like this: bounds=[(0, None) for b in range(N)].

# XC is used to impose other constraints, in addition to the bounds.

# ret = minimize ( objective, start, bounds = B, constraints = XC )
# alpha = ret ['X']





# THE TASK

#   1.     suitable kernel function - The kernel function takes two data points as arguments and
#           returns a “scalar product-like” similarity measure; a scalar value. Start with the
#           linear kernel which is the same as an ordinary scalar product, but also explore the
#           other kernels in section 3.3.
#   2.     implement the objective function mentioned above    (equation 4, that crazy looking thing),
#          only receives the alpha vector as parameter. (hint. use global variables for other things it needs
#   3.     Implement te function zerofun. This function should implement the equality constraint of (10). Also here,
#          you can make use of numpy.dot to be efficient.
#   4.     Call minimize.  Make the call to minimize as indicated in the code sample above. Note
#          that minimize returns a dictionary data structure; this is why we must
#          must use the string 'x' as an index to pick out the actual α values.
#          There are other useful indices that you can use; in particular, the index
#          'success' holds a boolean value which is True if the optimizer actually
#          found a solution.
#   5.     Extract the non-zero α values.   If the data is well separated, only a few of the α values will be non-zero.
#          Since we are dealing with floating point values, however, those that are
#          supposed to be zero will in reality only be approximately zero. Therefore,
#          use a low threshold (10−5 should work fine) to determine which are to be
#          regarded as non-zero.
#          You need to save the non-zero αi’s along with the corresponding data
#          points (⃗xi) and target values (ti) in a separate data structure, for instance
#          a list.
#   6.    Calculate the b value using equation (7) Note that you must use a point on the margin. This corresponds to a
#         point with an α-value larger than zero, but less than C (if slack is used).
#   7.    Implement the indicator function. Implement the indicator function (equation 6) which uses the non-zero
#          αi’s together with their ⃗xi’s and ti’s to classify new points.



# HINT for objective function:
# precompute a matrix P i,j = ti tj K(⃗xi, ⃗xj )

# indices i and j run over all the data points. Thus, if you have N
# data points, P should be an N × N matrix.

# This matrix should be computed only once, outside of the function objective. Therefore,
# store it as a numpy array in a global variable.

# Inside the objective function, you can now make use of the functions
# numpy.dot (for vector-vector, vector-matrix, and matrix-vector multiplications),
# and numpy.sum (for summing the elements of a vector).
# This is much faster than explicit for-loops in Python





#            TASK 1:  Create Kernel Functions

# Variables to help with Kernels
p = 3               # p for poly
sig = 1.5           # sig for sigma

# Linear K ( vector X, vector Y )  =  Transpose of Vector X * Vectory Y

def kernal_linear ( x, y ):
     theOutput = numpy.dot( numpy.transpose( x ), y )
     return theOutput

def kernal_poly ( x, y ):
    theOutput = (  numpy.dot( numpy.transpose( x ), y) + 1 ) ** p #p from above
    return theOutput

def kernal_rbf ( x, y ):
    theOutput = math.exp(  - (  ( numpy.linalg.norm(x - y) ) ** 2 / (2 * sig ** 2) )  ) #sig from above
    return theOutput


#            TASK 2: implement the Objective function mentioned above.
#           (equation 4, that crazy looking thing)
#           It only receives the alpha vector as parameter.
#           (hint. use global variables for other things it needs)
#           It takes vector ⃗α as argument and returns a scalar value
#          it implements the expression that should be minimized (equation 4 in the pdf).


# Thinking about the Indicator function.
# since we have training data thats labeled +1 and -1
#    the product of the testing 'label' and the predicted 'label' will be + if classified correctly, - if not
#    it "indicates" what the data is, +1 or -1.  We've constrained the support vectors to have minimum of -1 and +1 distance to the center of that dividing line.

def objective ( x ):
    scalarValue = 'some value, undetermined thus far'
    return scalarValue


# PROCEDURAL AND DRIVING CODE

y = numpy.array(  [2.0, -3.0, 1.0] )

x = numpy.array ( [2.50, -3.25, 1.00] )


print(   kernal_rbf(x, y)  )









# PLOTTING THE ABOVE DATA

plt.plot( [ p[0] for p in classA], [ p[1] for p in classA], 'b. ' )
plt.plot( [ p[0] for p in classB], [ p[1] for p in classB], 'r. ' )

plt.axis('equal')           # Force same scale on both axes
plt.savefig('svmplot.pdf')  # Save a copy in a file
plt .show()                 # Show the plot on the screen


print ( " the size of the targets list is: ", len(targets) )