######################################################################
# DOCUMENTATION / README
######################################################################

# Computes and analyzes cokernels related to Adams operations on certain space and spectra.
# For more information, see README.md

# Copyright (C) 2021 Eduardo Fischer

# This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License version 3
#as published by the Free Software Foundation. See LICENSE.
# Alternatively, see https://www.gnu.org/licenses/.

# This program is distributed in the hope that it will be useful,
#but without any warranty; without even the implied warranty of
#merchantability or fitness for a particular purpose.

######################################################################
# CODE: IMPORTS AND GLOBAL VARIABLES
######################################################################

# Import time module
import time as time

# Import to count the number of CPU cores for function find_cpu_cound
# Sadly it is not possible to import in such a way psutil.cpu_count is usable
#unless we import the whole psutil, which we prefer not to do
from psutil import cpu_count as cpu_count_from_psutil

# Some functions will use multiprocessing over this number of cores (unless a request bypasses this number)
NUMBER_OF_CORES = 6

# Name of the variables for display. The default is 'beta_'
# beta_ was the first value ever, and likely the canonical one
# Many variables are very often named with reference to "beta_"
# And it is also reflected on comments and docstrings
SYMBOL_FOR_GENERATORS = 'beta_'

# This is done to make the code work with Python 2 and 3
# In Python 2 (and before Python 3.3), u'example \u2295' is of class unicode, which has issues with str(). '\u2295' is also poor
# In Python 3 (starting at Python 3.3), u'example \u2295' is of class str, and prints well. We don't even need to start with u''
# We decide to use + instead of the direct sum symbol, u'\u2295', in Python 2
# This way, we will always have a str object, which can be nicely concatenated to other strings
# So, we establish a notion to trigger one way or the other
try:
  DIRECT_SUM_SYMBOL = str(u'\u2295')
  # If this works, it is Python 3. If it fails, it is Python 2, and we replace it by '+'
except UnicodeEncodeError:
  DIRECT_SUM_SYMBOL = '+'

# Dict of possible things that grow for our study for easier renaming
# They appear as the first argument of study_cokernels, but we don't really need the dict for the purposes there
# Right now no abbreviations; names are short enough
POSSIBLE_GROWTH = {'p': 'p',
    'k': 'k',
    'max_j': 'max_j',
    'max_ordinal': 'max_ordinal',
    'd': 'd',
    'fixed_degree': 'fixed_degree'}

# Dict of possible actions for our study
# If string is commanded as output, then it will be indexed by the variable parameters
# Maybe the indexed could be thought of as being a verbose option, thus explaining the two-letter abbreviation
POSSIBLE_ACTIONS = {'no': 'nothing',
    'pc': 'pure_cokernels', # cokernels as abelian groups with orders and generators
    'po': 'pure_orders', # cokernels given by the orders of the cyclic groups making it up
    'to': 'total_order', # order of the cokernel, good if finite
    'mo': 'max_of_orders', # max among the orders of the cyclic groups making the cokernel up
    'pl': 'pure_logs', # logarithm of the orders of the cyclic groups making up the cokernel
    'sl': 'sum_of_logs', # logarithm of the orer of the cokernel
    'ml': 'max_of_logs', # max logarithm of orders of cyclic groups making up the cokernel
    'as': 'assure_sum_increases', # boolean for testing if the total order increases
    'am': 'assure_max_increases', # like option 'as' but for the max of the orders of the cyclic groups
    'al': 'assure_list_increases'} # like option 'as' but for lists as in NonDecreasingIntegerList

######################################################################
# CODE: CLASSES
######################################################################

# Class BasedMatrix
# Stores a linear transformation as the product of three matrices, A*M*B
# A is change of base on the codomain, B is change of base on the codomain
# Main method is diagonalization of M (to obtain kernel and cokernel)
# Two subclasses: one for coefficients in Z_p^hat, and another for Z/p^k
# The subclasses may have exclusive methods, like make_M_diagonal, which is only for Z_p^hat
class BasedMatrix(object):
  r'''
  General class for matrices with preservation of domain and codomain base changes.
  
  This is accomplished through the attributes `M`, `A` and `B` of `self`.
  The product A*M*B is kept constant. If we diagonalize the matrix, `M`
  becomes diagonal and `A`and `B` are responsible for the base changes.
  
  For this class to work it needs some arguments obtainable from methods, generally `__init__`
  in subclasses. Not all are strictly necessary.
  
  `self.n`
  `self.p`
  `self.original_ring`
  `self.original_matrix_space`
  `self.qq_matrix_space`
  `self.invertible_matrix_space`
  `self.M`
  `self.A`
  `self.B`
  `self.A_inv`
  `self.B_inv`
  `self.AMB`
  `self.which_matrices_updated`
  `self.is_M_guaranteed_diagonal`
  '''

  # Method __str__()
  # Expected result is a single string with everything one wants outputed
  # Slight modifications depending on subclass
  def __str__(self):
    r'''
    Magic method. Returns a human-readable representation of the instance.
    '''
    return_string = ''
    if isinstance(self, BasedMatrixZPHat):
      return_string += 'All elements below in MatrixSpace(Z_{}^hat, {}, sparse = False)\n'.format(self.p, self.n)
    elif isinstance(self, BasedMatrixZPK):
      return_string += 'All elements below in MatrixSpace(Z/({}^{}), {}, sparse = False)\n'.format(self.p, self.k, self.n)
    else:
      # We should never fall here, but at the same time we don't want to code an Error/Exception for this at this place
      # So we just continue the code, no problems.
      pass
    return_string += 'Matrix A (change of base on codomain) is:\n'
    return_string += str(self.matrix_display('A'))
    return_string += '\nMatrix M (matrix for bases given by A and B) is:\n'
    return_string += str(self.matrix_display('M'))
    return_string += '\nMatrix B (change of base on domain) is:\n'
    return_string += str(self.matrix_display('B'))
    return_string += '\nLinear transformation on canonical base is given by the product A*M*B:\n'
    return_string += str(self.matrix_display('AMB'))
    return return_string

  # Method __repr__()
  # A*M*B should never change after initialization, since we always multiply by J and J_inv together
  # Of course user has the power to change A, M, or B separately. But we absolutely shouldn't.
  # We could use _A, _B, _M to denote private attributes, but we won't.
  def __repr__(self):
    r'''
    Magic method. Returns an unambiguous representation of the instance.
    '''
    # Creating a list of tuples:
    AMB_as_list_of_tuples = list(self.matrix_display('AMB'))
    # Making it a list of lists
    AMB_as_list_of_lists = [list(each_tuple) for each_tuple in AMB_as_list_of_tuples]
    # This is likely broken. Don't think it really works for both subclasses...
    return '{}({}, {}, {})'.format(self.__class__.__name__, self.p, self.n, AMB_as_list_of_lists)
    # If you choose str(type(self)) you get "Instance" instead of the class name
    # If you choose return (self.p, self.n, '\n', self.AMB) you don't get good line breaks.

  # Method __mul__()
  # Implements multiplication (on the left) by a number, number * instance
  # Effect is on M only, keeping A and B intact (and size and original ring and everything)
  # Alter self, but rather returns self
  # The most Pythonic and functional way of doing this would be not to alter self
  # However, there are problems with __init__().
  # Would we __init__() again and have to diagonalize again, wasting time?
  # There isn't, at the moment, a way of initializing the matrix with self.A and self.B not being the identity
  # We could chance __init__(), but we will rather have this method not be functional.
  # Similar observations for __rmul__() and __imul__()
  def __mul__(self, number):
    r'''
    Magic method. Outputs the result of multiplication by the argument on the left.
    '''
    assert number in self.original_ring
    self.M = number*self.M
    self.AMB = number*self.AMB
    return self

  # Method __rmul__
  # Implements multiplication (on the right) by a number, instance * number
  # Refers to the one in the other left
  def __rmul__(self, number):
    r'''
    Magic method. Outputs the result of multiplication by the argument on the right.
    '''
    return self.__mul__(number)
    
  # Method __imul__
  # Implements multiplication by a number via assignment, instance *= number
  def __imul__(self, number):
    r'''
    Magic method. Outputs the result of multiplication by the argument via assignment.
    '''
    return self.__mul__(number)

  # Method matrix_display
  # Makes one of the matrices A, M, B or AMB, more displayable
  # Right now, it displays as QQ matrix if possible, otherwise self.original_ring
  def matrix_display(self, letter):
    r'''
    Returns one of the matrices in self in a visually pleasant way.
    '''
    # We ensure it makes sense.
    assert letter in ['A', 'M', 'B', 'AMB', 'A_inv', 'B_inv'], 'Invalid choice for letter'
    # We ensure the attribute is updated
    if letter in ['A', 'B', 'A_inv', 'B_inv']:
      self.update_matrix(letter)
    # We prefer to display as rational matrix (at the moment). If unavailable, we go with original_ring
    try:
      return self.matrix_qq(letter)
    except TypeError:
      return getattr(self, letter)

  # Method matrix_qq
  # Gets one of the matrices, A, M, B or AMB, and displays it with coefficients in QQ
  def matrix_qq(self, letter):
    r'''
    Returns one of the matrices in self with rational coefficients, if possible.
    '''
    # We ensure it makes sense.
    assert letter in ['A', 'M', 'B', 'AMB', 'A_inv', 'B_inv'], 'Invalid choice for letter'
    # We ensure the attribute is updated
    if letter in ['A', 'B', 'A_inv', 'B_inv']:
      self.update_matrix(letter)
    # We make it in a rational matrix. It works only for some instances, but the ones we typically use
    # If it doesn't, an error to appear is normal and expected
    return self.qq_matrix_space(getattr(self, letter))

  # Method is_M_diagonal
  # Determines if M is diagonal by examining its entries, returning the answer and setting the correct flag, self.M_guaranteed_diagonal
  # We do the expected way, with breaks to not test unnecessarily
  def is_M_diagonal(self, believe_in_M_guaranteed_diagonal = True):
    r'''
    Returns whether self.M is a diagonal matrix or not.
    '''
    if believe_in_M_guaranteed_diagonal and self.M_guaranteed_diagonal:
      return self.M_guaranteed_diagonal
    else:
      proved_not_diagonal = False
      for row in range(self.n):
        for column in range(self.n):
          if row != column:
            if self.M != 0:
              proved_not_diagonal = True
              break
        if proved_not_diagonal == True:
          break
      # We have just proved the following
      self.M_guaranteed_diagonal = not(proved_not_diagonal)
      return self.M_guaranteed_diagonal

  # Method interact_with_vector
  # Used to compute the pre-image of the direct image a (column) vector in the codomain.
  # It assumes we know the state we're dealing with. That's why it uses self.M by default (instead of AMB).
  # It has an option original_M, that, which set to true, computes pre-image of original M, which is always stored as self.AMB
  # We allow for many rings. If we want the computation in Z_p_hat we can't allow p in the denominator.
  def interact_with_vector(self, given_vector, direct_or_pre, move_to_qq = False, use_original_M = False, transpose = False):
    r'''
    Returns result of image or pre-image of a vector by self.M. If not possible, use pre-image of multiple of vector.
    '''
    # First we make the vector into a Sage vector (if it wasn't already)
    vector_as_vector = vector(given_vector)
    # Then we pick the matrix we want to use. The original M is stored as self.AMB
    # We make then rational first
    if use_original_M == True:
      pre_matrix_used = self.matrix_qq('AMB')
    else:
      pre_matrix_used = self.matrix_qq('M')
    # Now we ask if we need to transpose
    if transpose:
      matrix_used = pre_matrix_used.transpose()
    else:
      matrix_used = pre_matrix_used
    # Now we call format_direct_or_pre
    # We compute using the rational matrix. We later can work out the p part if p is not invertible.
    formatted_direct_or_pre = format_direct_or_pre(direct_or_pre)
    if formatted_direct_or_pre == 'direct':
      relevant_vector = matrix_used*vector_as_vector
    else:
      # We need to write some code for the case that over QQ the matrix isn't invertible
      # If it has no pre-image over QQ, it won't have over Z_p_hat
      try:
        relevant_vector = matrix_used.solve_right(vector_as_vector)
      except ValueError:
        # Let's establish that the only multiple in the image is 0 times the vector.
        # Kind of unorthodox, but probably the best...
        return (0, [0]*self.n)
    # If we allow p in the denominator, we return a tuple (multiple, image_of_multiple)
    if move_to_qq:
       # The multiple is 1 and image_of_multiple
      return (0, list(relevant_vector))
    # If we don't, we will want to find a power of p such that that power times the vector is in fact valid
    else:
      # Now we do computations in self.original_ring, typically Z_p_hat
      # If given_vector had a pre-image over QQ, maybe only a power of p times the vector will be in the image for Z_p_hat
      # We are determined to find that multiple and to output the multiple and the vector times that multiple
      # If any denominator is a multiple of p, we multiply by that power to ensure all vector entries belong to Z_p^hat
      necessary_power_of_p = 1
      minimal_evaluation_among_entries = min([self.q_p_hat(entry).ordp() for entry in relevant_vector])
      if minimal_evaluation_among_entries < 0:
        necessary_power_of_p = self.p**(-minimal_evaluation_among_entries)
        relevant_vector *= necessary_power_of_p
      # We will return a tuple of the final vector and the relevant multiple of p.
      return (necessary_power_of_p, list(relevant_vector))

  # Method prepare_canonical_base
  # To be used in codomain and domain alike
  # canonical_base should be always given as a list of objects. Those objects will be turned into strings.
  def prepare_canonical_base(self, canonical_base = None):
    r'''
    Prepares a canonical base for use in self.
    '''
    # Canonical base can be given as beta_1, beta_2, beta_1^2, ... output by method currently called produce_first_places
    # That is, the degrevlex order on a polynonial ring with finitely many variables
    # If None is given, we set it to e_1, ... e_n as a list
    # Useful to output text too.
    if canonical_base == None:
      # We start at e_1 independently if we are including beta_0 (which isn't even available to self nor to the method)
      # Not exactly Pythonic, but in Mathematics we often start counting at 1
      preparing_canonical_base = ['e_{}'.format(count+1) for count in range(self.n)]
    # If something is given, we will work on it if needed.
    else:
      # If given a tuple, we change it to a list format. Matter of preference. Let's start by renaming for our convenience.
      # We will uniformize it as list of strings int he following way
      preparing_canonical_base = [str(item) for item in canonical_base]
    return preparing_canonical_base

  # Method get_base_codomain
  # Returns a list of lists, each list being a codomain base vector in the (for now abstract) canonical base
  # This uses A in A*M*B
  # (This will produce a list which is a base of the codomain as Q-vector space. Its span over Z_p_hat may or may not be all.)
  def get_base_codomain(self):
    r'''
    Returns a base for the codomain of self as a list of lists.
    '''
    # We initialize a list of size self.n, where each object is a list of size self.n
    list_of_lists = [[0]*self.n for count in range(self.n)]
    # We verify we have self.A correctly updated
    self.update_matrix('A')
    # Now we modify the elements as needed. Now every element in each list (i.e. row) of the bigger list will be a Sage rational number.
    for base_vector_place in range(self.n):
      for each_coordinate_in_canonical_basis in range(self.n):
        # Note previously we had self.matrix_display('M') which was way costlier in transforming Z_p_hat -> QQ than the following
        list_of_lists[base_vector_place][each_coordinate_in_canonical_basis] = QQ(self.A[each_coordinate_in_canonical_basis, base_vector_place])
    # If we wanted, we could create a list of tuples by doing:
    #list_of_tuples = []
    #for base_vector in list_of_lists:
    #  base_vector_as_tuple = tuple(base_vector)
    #  list_of_tuples.append(base_vector_as_tuple)
    return list_of_lists

  # Method get_base_domain
  # This uses B (more specifically B_inv) in A*M*B
  # This will produce a list which is a base of the domain as Q-vector space. Its span over Z_p_hat may or may not be all.
  # Returns a list of lists, each list being a domain base vector in the (for now abstract) canonical base
  def get_base_domain(self):
    r'''
    Returns a base for the domain of self as a list of lists.
    '''
    self.make_M_diagonal()
    # We update B_inv to make sure the attribute holds the correct matrix
    self.update_matrix('B_inv')
    # We test to see if everything is really all right with our matrix B_inv
    #if type(self.B_inv) != sage.matrix.matrix_rational_dense.Matrix_rational_dense:
      #print('Warning: inverse of matrix B should be rational, but it isn\'t. Things may fail.')
    # Now we copy from get_base_codomain
    list_of_lists = [[0]*self.n for count in range(self.n)]
    for base_vector_place in range(self.n):
      for each_coordinate_in_canonical_basis in range(self.n):
        list_of_lists[base_vector_place][each_coordinate_in_canonical_basis] = QQ(self.B_inv[each_coordinate_in_canonical_basis, base_vector_place]) 
    return list_of_lists

  # Method print_base_codomain
  # Allows printing of the codomain base. Allows options and some formatting, plus option for canonical base.
  # output_as == 'list' gives list of strings, output_as == 'string' gives it as multi-line string
  def print_base_codomain(self, canonical_base = None, output_as = 'list'):
    r'''
    Returns a base for the codomain of self as list of strings or multi-line string.
    '''
    prepared_canonical_base = self.prepare_canonical_base(canonical_base) # This is a list of strings
    list_of_base_vectors = self.get_base_codomain() # This is a list of lists
    # If we want to return a list of strings instead of printing to the console, we start by creating a list of empty strings
    # Even if not, it is convenient for printing if it's what we want to do it right now.
    list_of_strings = ['']*(self.n)
    for base_vector_place in range(self.n):
      # If printing base by this own method, right now, we would announce it like this:
      if format_output_as(output_as) == 'string':
        list_of_strings[base_vector_place] += 'The {} codomain base vector is '.format(right_st_nd_rd_th(base_vector_place+1))
      # Let's add a Boolean variable to know when to add a ' + ' to the string. It is False only for the first piece added to the string. Reset each base_vector_place.
      should_add_plus = False
      for count in range(self.n):
        # We will add to the string only the nonzero coordinates of the base vector
        if list_of_base_vectors[base_vector_place][count] != 0:
          # Now we decide if we add ' +'. Since Sage already include minuses, only ' ' will be added to negative numbers
          if should_add_plus == True:
            if list_of_base_vectors[base_vector_place][count] >= 0:
              # Single space plus a +
              list_of_strings[base_vector_place] += ' +'
            else:
              # Single space
              list_of_strings[base_vector_place] += ' '
          list_of_strings[base_vector_place] += str(list_of_base_vectors[base_vector_place][count])+'*'+prepared_canonical_base[count]
          # Now that at least one is nonzero, we need to start adding pluses between the coordinates
          should_add_plus = True
    # Now we have a finalized list of self.n strings, and will decide what to do with them.
    # Note that the ' codomain base vector is ' string only appears if function is asked to output as a string
    # If we want to print now, we return a string well-formatted for printing
    if format_output_as(output_as) == 'string':
      printing_string = '\n'.join([list_of_strings[count] for count in range(self.n)])
      return printing_string
    # If we don't want to print now, we output a list of strings. This should be the deault option.
    else:
      return list_of_strings

  # Method print_base_domain
  # Allows printing of the domain base. Allows options and some formatting, plus option for canonical base.
  def print_base_domain(self, canonical_base = None, output_as = 'list'):
    r'''
    Returns a base for the domain of self as list of strings or multi-line string.
    '''
    # This code is a copy of print_base_codomain(), except for string changes.
    # Because of it, no further comments.
    prepared_canonical_base = self.prepare_canonical_base(canonical_base)
    list_of_base_vectors = self.get_base_domain()
    list_of_strings = ['']*(self.n)
    for base_vector_place in range(self.n):
      if format_output_as(output_as) == 'string':
        list_of_strings[base_vector_place] += str(base_vector_place+1)+right_st_nd_rd_th(base_vector_place+1)+' domain base vector is '
      should_add_plus = False
      for count in range(self.n):
        if list_of_base_vectors[base_vector_place][count] != 0:
          if should_add_plus == True:
            if list_of_base_vectors[base_vector_place][count] >= 0:
              list_of_strings[base_vector_place] += ' +'
            else:
              list_of_strings[base_vector_place] += ' '
          list_of_strings[base_vector_place] += str(list_of_base_vectors[base_vector_place][count])+'*'+prepared_canonical_base[count]
          should_add_plus = True
    if format_output_as(output_as) == 'string':
      printing_string = '\n'.join([list_of_strings[count] for count in range(self.n)])
      return printing_string
    else:
      return list_of_strings

  # Method produce_cokernel
  # Produces the cokernel of the matrix over a ring as a list of abelian groups
  # Matrix is self.M, and the ring is self.Z_p_hat
  # Name is only used for output_as == 'object' and 'string', completely ignored for 'list'
  # At the moment, we won't unify this with "produce_cokernel_orders" because they use completely different procedures and produce meaningfully different outputs
  def produce_cokernel(self, canonical_base = None, output_as = 'object', cokernel_name = 'Generic cokernel'):
    r'''
    Produces the cokernel of self.M using the codomain base given by self.A.
    
    EXAMPLE:
    
    If self.M is
    
    [0 0 0]
    [0 2 0]
    [0 0 5]
    
    and the self.original_ring is Zp(5) = Z_5^hat then the cokernel,
    mathematically, should be isomorphic to Z_5_hat + Z/5Z.
    In this implementation, it will be output as an instance of
    FinitelyGeneratedAbelianGroup corresponding to the group ZZ + Z/5Z.
    '''
    # Then we diagonalize self.M to allow us to read the cokernel.
    self.make_M_diagonal()
    # We then get the base of the codomain, which print_base_codomain formats as a list of strings
    # Included in this step, we have a call of the method prepare_canonical_base() which prepares the base
    base_codomain = self.print_base_codomain(canonical_base)
    # Cokernel should be (Z/p^kZ)*e_k for every element in the diagonal which isn't 1
    # We're ignoring ZZ<beta_0>, as it doesn't come from this class, but rather from other considerations
    # We want to produce a list of groups, and we start with an empty list
    list_of_groups = []
    for count in range(self.n):
      # In p-completed matrices, there is only a nontrivial summand for the cokernel when the diagonal entry is a power of self.p higher than 1
      # (When it's exactly 1, it is a Z/1Z summand, which we will supress from the results.)
      # If we have a diagonal entry 0, then we have order oo and a summand ZZ or Z_p^hat (depending on the point of view)
      # The diagonal entry 0 can happen in two cases. First is for an instance of BasedMatrixZPK, the other is then d is negative.
      # Each group is a tuple, characterized by group[0] being its order, and group[1] being a generator (as string)
      if self.M[count, count] != 1:
        if self.M[count, count] == 0:
          if isinstance(self, BasedMatrixZPK):
            # Let's add a ZZ to force this to be Sage integer instead of Sage rational
            appending_tuple = (ZZ(self.p**self.k), str(base_codomain[count]))
            list_of_groups.append(appending_tuple)
          else:
            # In this case, in absence of torsion, we believe it should be a copy of ZZ or Zphat
            # We have no means to effectively test torsion other that isinstance(self, BasedMatrixZPK)
            # And so we will be very happy in simply write 0 (for the order, meaning ZZ or Zphat, as it's done everywhere)
            appending_tuple = (ZZ(0), str(base_codomain[count]))
            list_of_groups.append(appending_tuple)
        else:
          # Let's add a ZZ to force this to be Sage integer instead of Sage rational
          appending_tuple = (ZZ(self.M[count, count]), str(base_codomain[count]))
          list_of_groups.append(appending_tuple)
    # If output_as == list, return a list of tuples, (order as a number, generator as a string).
    if format_output_as(output_as) == 'list':
      return list_of_groups
    else:
      fgagroup = FinitelyGeneratedAbelianGroup(cokernel_name, list_of_groups)
      if format_output_as(output_as) == 'object':
        return fgagroup
      elif format_output_as(output_as) == 'string':
        return str(fgagroup)

  # Method produce_kernel
  # Produces the kernel of a square matrix over a ring
  def produce_kernel(self, canonical_base = None, output_as = 'object', kernel_name = 'Generic kernel'):
    r'''
    Produces the kernel of self.M using the codomain base given by self.B.
    '''
    # If the matrix is invertible over Q, there is no kernel over self.Z_p_hat (we denote this trivial group by 1)
    # There is also no kernel over Z/p^k either, as no vector has order a multiple of p
    # Of course, if the instance comes from Adams operations, beta_0 was likely excluded.
    # In that case, a copy of ZZ<beta_0> should be added to the kernel (technically Z_p_hat<beta_0>). But not within this instance method.
    # To be consistent with method produce_cokernel(), we will adapt its syntax
    # In the future, if this method ever becomes useful, add option for outputting a FinitelyGeneratedAbelianGroup
    # To read the kernel, it suffices to create a Z_p_hat (denoted ZZ as FinitelyGeneratedAbelianGroup)
    #for each vector in the domain basis having a 0 in the diagonal of M. If diagonal element != 0, not on kernel.
    # Repeating: We will call ZZ that copy of Z_p_hat
    # The kernel will always be torsion-free, a collection of ZZ's or Z_p_hat's
    self.make_M_diagonal()
    list_of_groups = []
    # We will have self.n base vectors, each represented by a string
    list_of_base_vectors_as_strings = self.print_base_domain()
    for count in range(0, self.n):
      if self.M[count, count] == 0:
        # Recall that a summand of a group is given as a tuple, 0 representing ZZ (Z_p_hat technically, but we are happy with ZZ)
        appending_tuple = (0, list_of_base_vectors_as_strings[count])
        list_of_groups.append(appending_tuple)
    formatted_output_as = format_output_as(output_as)
    if formatted_output_as == 'list':
      # That is, a list of summands, each (order, generator)
      return list_of_groups
    else:
      fgagroup = FinitelyGeneratedAbelianGroup(kernel_name, list_of_groups)
      if format_output_as(output_as) == 'object':
        return fgagroup
      elif format_output_as(output_as) == 'string':
        return str(fgagroup)
      else:
        return fgagroup

  # Method produce_cokernel_orders
  # Uses the elementary_divisors() method for Sage matrices
  # Gets the order of the groups produces in produce_cokernel, but no generators
  # The order is a list of numbers, 0 meanings being Z and n > 0 meaning Z/nZ
  # The advantage is that it is faster
  def produce_cokernel_orders(self, output_as = 'object', cokernel_name = 'Generic cokernel'):
    r'''
    Produces the orders of the cokernel of self.M.
    '''
    # First we use elementary_divisors() on self.M
    # Since we typically store self.M as matrix over QQ, we need to coerce it to Z_p^hat
    # (and the coercion works because every element is in the intersection of QQ and Z_p^hat)
    # Later considering moving the following to __init__
    list_of_orders = self.M.elementary_divisors()
    # Then we exclude the 1's (Z/1Z summands are uninteresting), and ensure we work with ZZ
    list_of_orders_without_trivial_groups = [ZZ(order) for order in list_of_orders if order != 1]
    # We typically start with the Z's with the produce_cokernel_method. Because of that, we will put the 0's first here too. Easier is:
    list_of_orders_without_trivial_groups.sort()
    # Since we don't have proper generators, we will leave it to __init__ of FinitelyGeneratedAbelianGroups to provide generic names
    # We will let  FinitelyGeneratedAbelianGroup.__init__ create the variable names by passing a list of empty strings
    list_of_empty_names = ['']*len(list_of_orders_without_trivial_groups)
    fgagroup = FinitelyGeneratedAbelianGroup(cokernel_name, list_of_orders_without_trivial_groups, list_of_empty_names)
    formatted_output_as = format_output_as(output_as)
    if formatted_output_as == 'list':
      return list_of_orders_without_trivial_groups
    elif formatted_output_as == 'string':
      return fgagroup.print_flexibly(printing_choice = 'orders_only')
    elif formatted_output_as == 'object':
      return fgagroup
    else:
      return fgagroup

  # Method produce_kernel_rank
  # In analogy to produce_cokernel_orders, but since the FinitelyGeneratedAbelianGroup stands for a free Z_p_hat-module, we adapt to rank
  # Technically there is a identification ZZ with Z_p_hat (in the kernel, we don't make it really really clear)
  def produce_kernel_rank(self, output_as = 'object', kernel_name = 'Generic cokernel'):
    r'''
    Returns the rank of the kernel of self.
    '''
    # Easiest way is to read the  number of zero elements on the diagonal of self.M
    # (Likely also possible with native Sage methods... but the way it is now is fine for us)
    self.make_M_diagonal()
    rank = 0
    for count in self.n:
      if self.M[count, count] == 0:
        rank += 1
    # Doesn't matter the output_as option, we simply output the number
    return rank

  # Method update_matrix
  # To guarantee we have the right matrix, even if it wasn't computed yet
  def update_matrix(self, letter):
    r'''
    Updates one among 'A', 'B', 'A_inv' and 'B_inv' of self to have its correct value.
    '''
    # We first ensure we have a valid matrix to update
    # Note that self.A, self.B, self.A_inv and self.B_inv are defined in all moments
    # Except that one might not be updated, but in this case its inverse is
    try:
      has_been_updated = self.which_matrices_updated[letter]
    except KeyError:
      raise ValueError('Can only update \'A\', \'B\', \'A_inv\' and \'B_inv\' through this method.')
    if has_been_updated:
      pass
    else:
      # We create a dictionary to write fewer lines of code
      inverse_of_each_letter = {'A': 'A_inv', 'B': 'B_inv', 'A_inv': 'A', 'B_inv': 'B'}
      setattrib(self, letter, inverse_of_each_letter[letter]**(-1))
    return 'Warning: update_matrix() works as procedure, not as function. It modifies the instance. Don\'t return it like you just did.'

######################################################################

# Subclass BasedMatrixZPHat
# Matrices have coefficients in Z_p^hat.
class BasedMatrixZPHat(BasedMatrix):
  r'''
  Subclass of BasedMatrix. To be used when matrix has coefficients in Z_p^hat.
  '''

  # Method __init__ initializes the instance of the class
  def __init__(self, given_p, given_n, given_M, transpose = False):
    r'''
    Magic method. Initializes an instance of the class.
    '''
    # Note n here is the dimension of the instance, and it has nothing to do with BU(n) or with n_bound
    # For our main use, it will typically be max_ordinal+1
    self.p = given_p
    self.n = given_n
    # Can later do localization or completion instead of QQ if appears necessary/better
    # Let's create nices-to-have to have connected to each instance.
    # Experimentation has shown that precision needs to grow with the size of the matrix to avoid PrecisionError
    # We scale it with self.n, and minimum is 1000. At this level it has almost no influence in the running time.
    # Since including a dimension d > 0 has the potential to cause issues, we will keep the precision relatively high.
    self.z_p_hat_precision = max(1000, 5*self.n)
    self.z_p_hat = Zp(self.p, self.z_p_hat_precision, 'capped-rel')
    self.q_p_hat = self.z_p_hat.fraction_field()
    self.z_p_hat_matrix_space = MatrixSpace(self.z_p_hat, self.n)
    self.q_p_hat_matrix_space = MatrixSpace(self.z_p_hat, self.n)
    # Let's keep it simple. Matrizes are rational (since we assume we are given a p-localized matrix.....)
    self.qq_matrix_space = MatrixSpace(QQ, self.n, sparse=False)
    # The following two attributes are to be worked with BasedMatrix
    self.original_ring = self.z_p_hat
    self.original_matrix_space = self.z_p_hat_matrix_space
    self.invertible_matrix_space = self.q_p_hat_matrix_space
    # Maybe do a variable type checking to allow higher variety of inputs as given_M
    # But the code below coerces it to the right matrix space
    if transpose:
      self.M = self.original_matrix_space(given_M).transpose()
    else:
      self.M = self.original_matrix_space(given_M)
    # (If it was invertible on Z_p_hat, then cokernel and kernel would be 0.)
    # Thie first iterations of this class only accepted invertible matrices. This is no longer the case.
    # A short way to get the identity matrix n by n
    # Could also do self.A = Matrix(QQ, self.n, matrix.identity(self.n))
    # But we'd rather have flexibility in the matrix space
    self.A = self.invertible_matrix_space.identity_matrix()
    self.B = self.invertible_matrix_space.identity_matrix()
    self.A_inv = self.invertible_matrix_space.identity_matrix()
    self.B_inv = self.invertible_matrix_space.identity_matrix()
    self.which_matrices_updated = {'A': True, 'B': True, 'A_inv': True, 'B_inv': True}
    # At this moment AMB it will be simply M, == but not id() ==. No method is allowed to change this product A*M*B.
    self.AMB = self.A*self.M*self.B
    # Let's setup this as a flag to never have to do the method make_M_diagonal twice
    # Most consistent way is to determine if diagonal by checking it on __init__(), then setting the flag self.M_guaranteed_diagonal
    self.M_guaranteed_diagonal = self.is_M_diagonal(believe_in_M_guaranteed_diagonal = False)

  # Method make_M_diagonal
  # We basically call smith_form()
  # This can also force all the entries of M to be 0 or p^k, for k>=0
  # In the case, they will also what are called the elementary_divisors()
  # Cokernel will be a direct sum of Ring/diagRing for each element diag of the diagonal
  def make_M_diagonal(self):
    r'''
    Modifies self.A, self.M and self.B so that self.M is a diagonal matrix, all while keeping their product constant.
    From another point of view, it finds bases such that self.AMB is a diagonal matrix.
    '''
    # For 1x1 matrices smith_form() appears to fail as probably a floating point problem for p-adics
    # So we will do it manually in this case
    if self.n == 1:
      only_element = self.M[0][0]
      # If it is 0 we don't do anything
      if only_element == 0:
        pass
      else:
        # We will change the matrix self.B (responsible for kernel) and keep self.A (responsible for cokernel) intact
        correct_power_of_prime = only_element.ordp(self.p)
        # correct_power_of_prime should be greater than 0 as we assume the matrix was in Z_p_hat
        assert correct_power_of_prime >= 0, 'Matrix should be have coefficients in Z_p_hat'
        self.M = self.z_p_hat_matrix_space(1, 1, self.z_p_hat(self.p**correct_power_of_prime))
        self.B = self.z_p_hat_matrix_space(1, 1, self.z_p_hat(only_element/(self.p**correct_power_of_prime)))
        # Note only_element/(self.p**correct_power_of_prime) is an invertible in Z_p^hat
        self.B_inv = self.z_p_hat_matrix_space(1, 1, self.z_p_hat((self.p**correct_power_of_prime)/only_element))
        # We mark the matrices which have been updated
        self.which_matrices_updated.update({'B': True, 'B_inv': True})
    else:    
      # If we already diagonalized M, there is no reason to do this again. We can detect it by flag or is_M_diagonal
      if self.is_M_diagonal(believe_in_M_guaranteed_diagonal = False):
        pass
      else:
        # Here we use the native smith_form() method
        # We need to do it with coefficients in Z_p^hat
        # Note smith_form(M) = smith_form(A*M*B) returns D, U, V with D = U*input*V
        # So D is the new M, U is A_inv, V is B_inv
        self.M, self.A_inv, self.B_inv = self.M.smith_form()
        # We update information on which matrices are updates
        self.which_matrices_updated.update({'A_inv': True, 'B_inv': True})
      # Let's use this flag to avoid repeating this process.
      self.M_guaranteed_diagonal = True
    # This method changes the state of self, more precisely of self.A, self.M and self.B
    # We could opt for return None, as there is no natural choice on whether we should return self.M or all of self.A, self.M and self.B
    # self.make_M_diagonal() is a procedure that alters the internal state, or attributes, of self, it isn't a function
    # We will even return a warning message about that
    return 'Warning: make_M_diagonal() works as procedure, not as function. It modifies the instance. Don\'t return it like you just did.'

  # Method modular_reduction
  # Used to get matrices modulo p^k
  # Always works as every coefficient, let it be in A, M, B or A*M*B, is is Z_p^hat
  # Output is another instance of this class (and so any method can be used)
  # This is possible as Z/p^k is a subset of Z_p^hat
  def modular_reduction(self, k):
    r'''
    Returns an instance in which the coefficients of self are reduced from Z_p^hat to Z/p^kZ.
    '''
    # We create instances of BasedMatrixZPK, except for k = 0 which means a copy of self (meaning no reduction modulo p**k)
    if k == 0:
      return self
    else:
      # We always want to start with a diagonalized matrix, so we don't bother diagonalizing each reduction
      self.make_M_diagonal()
      # Let's make sure we have the correct B_inv without computing them for each BasedMatrixZPK instance
      # (B_inv would be necessary in get_base_codomain. A_inv is normally not very useful but we pass it forward anyway.)
      self.update_matrix('B_inv')
      # Now we initiate an instance of BasedMatrixZPK
      new_instance = BasedMatrixZPK(self.p, k, self.n, self.A, self.M, self.B, self.AMB, self.A_inv, self.B_inv)
      return new_instance

  # Method range_of_modular_reductions
  # Gets reductions modulo p^k for k between 1 and max_k
  # Output is a list of size max_k+1. In position 0, we keep the original instance of BasedMatrixZPHat
  # In position k (1 <= k <= max_k), we have the corresponding reductions modulo p^k as an instance of BasedMatrixZPK
  def range_of_modular_reductions(self, max_k):
    r'''
    Returns a list of instances in which the coefficients of self are reduced from Z_p^hat to Z/p^kZ.
    '''
    # We always want to start with a diagonalized matrix, so we don't bother diagonalizing each reduction
    self.make_M_diagonal()
    # Let's make sure we have the correct A_inv and B_inv computing it for each k later
    self.update_matrix('B_inv')
    # Now we create a list with the help of modular_reduction()
    list_of_new_instances = [self.modular_reduction(k) for k in range(0, max_k+1)]
    return list_of_new_instances

######################################################################

# Subclass BasedMatrixZPK
# Matrices have coefficients in Z/p^k
# We expect to get them as approximation to a matrix with coefficients in Z_p^hat
# As so, we assume matrix is already diagonalized, and everything is all right.
# There are fewer options (tranpose was removed), and some things are initialized more directly.
class BasedMatrixZPK(BasedMatrix):
  r'''
  Subclass of BasedMatrix. To be used when matrix has coefficients in Z/(p^k)Z.
  '''

  # Method __init__ initializes the instance of the class
  def __init__(self, given_p, given_k, given_n, given_A, given_M, given_B, given_AMB, given_A_inv, given_B_inv):
    r'''
    Magic method. Initializes an instance of the class.
    '''
    # We establish the previous matrix space (where the matrices come from) using parent() on given_M
    # (In general, an instance of this class is derived from another subclass of BasedMatrix)
    self.previous_matrix_space = given_M.parent()
    # We now do the verifications. They are not that costly, so we do it for peace of mind.
    # The following is adapted from BasedMatrix and BasedMatrixZPHat.
    self.p = given_p
    self.n = given_n
    self.k = given_k
    # Now we define our matrix space, and coerce the matrix to have their coefficients in Integers(self.p**self.k) = Z/p^k
    self.z_p_k = Integers(self.p**self.k)
    self.z_p_k_matrix_space = MatrixSpace(self.z_p_k, self.n, sparse=False)
    # The following two attributes are to be worked with BasedMatrix
    self.original_ring = self.z_p_k
    self.original_matrix_space = self.z_p_k_matrix_space
    # Now the usual definitions
    # If any of them can't be converted to matrices in self.z_p_k_matrix_space
    #we simply forget information about the bases for domain and codomain (that is, A, B, A_inv, B_inv are made identity)
    #and keep M intact since it is very likely diagonal the moment this is called (and thus AMB = M)
    self.M = self.original_matrix_space(given_M)
    try:
      self.A = self.original_matrix_space(given_A)
      self.B = self.original_matrix_space(given_B)
      self.AMB = self.original_matrix_space(given_AMB)
      self.A_inv = self.original_matrix_space(given_A_inv)
      self.B_inv = self.original_matrix_space(given_B_inv)
      # We use a dict to control which variables are updated
      self.which_matrices_updated = {'A': True, 'B': True, 'A_inv': True, 'B_inv': True}
    # The conversion Z_p^hat to Z/(p^k)Z should always work. If it doesn't (for example, elements of Q_p^hat)
    #there can be many errors, like for example ZeroDivisionError. So we don't specify the error in this try/except
    except:
      self.A = self.original_matrix_space.identity_matrix()
      self.B = self.original_matrix_space.identity_matrix()
      self.A_inv = self.original_matrix_space.identity_matrix()
      self.B_inv = self.original_matrix_space.identity_matrix()
      # We avoid self.AMB = self.M because only creating a reference can cause problems
      self.AMB = self.original_matrix_space(given_M)
      self.which_matrices_updated = {'A': True, 'B': True, 'A_inv': True, 'B_inv': True}
    # We leave code to verify all products are correct (we typically assume them correct on this very __init__)
    #assert self.A * self.M * self.B == self.AMB, 'The product of A, M and B should be AMB'
    #assert self.A*self.A_inv == self.previous_matrix_space.identity_matrix(), 'A and A_inv should be each other\' inverses'
    #assert self.B*self.B_inv == self.previous_matrix_space.identity_matrix(), 'B and B_inv should be each other\' inverses'
    # In most uses, M is supposed to be initiated as diagonal. Either way, we set the flag correctly.
    self.M_guaranteed_diagonal = self.is_M_diagonal(believe_in_M_guaranteed_diagonal = False)

  # Method make_M_diagonal
  # This is only to occupy the namespace, as the matrices are supposed to be diagonal already (from the way they're typically created)
  # Not including it throws errors
  def make_M_diagonal(self):
    r'''
    Does nothing. It exists to occupy the namespace.
    '''
    pass
    return 'Warning: make_M_diagonal() works as procedure, not as function. It modifies the instance. Don\'t return it like you just did.'

# Class ElementHomologyBU
# One instance is an element of K_0(BU(n)) for a specific n
# They are generated by monomials on the variables \beta_1, \beta_2, ...
class ElementHomologyBU(object):
  r'''
  Represents an element of the K-homology of BU.
  
  Construction data can be given in three main ways.
  i) a polynomial as an instance of self.rat_ring
  ii) a polyomial as a string recognizable by self.rat_ring
  iii) a dictionary which assigns to a tuple representin each monomial (a_1, ... ,a_{j_bound}) its coefficient
  
  Others can be accepted as long as they can be coerced by self.rat_ring using SageMath native methods.
  '''

  # Method __init__ initializes the instance of the class
  # p is mostly invisible but ok
  def __init__(self, given_p, given_n_bound, given_j_bound, given_data):
    r'''
    Magic method. Initializes an instance of the class.
    '''
    # The following say that it is an element of (KU_{self.p}^hat)_0(BU(self.n_bound))
    # Or in another K_{2d}... it is impossible to detect
    # Technically we could try to, in string_of_good_variables(1, self.j_bound+1), include u^d
    # But it is too much work to make it work coherently. Let's just say beta_1, beta_2, and so on.
    # self.j_bound means that every beta_j has j <= j_bound
    self.p = given_p
    self.n_bound = given_n_bound
    self.j_bound = given_j_bound
    # Let's define self.rat_ring right away since we always need it.
    self.rat_ring = PolynomialRing(QQ, self.j_bound, string_of_good_variables(1, self.j_bound+1), order='degrevlex')
    # Now we build the data the correct way
    # If starting with string, dictionary or polynomial as Sage object, we coerce it into belonging to self.rat_ring
    if isinstance(given_data, (str, dict, sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular)):
      try:
        # We simply try to make it into a polynomial
        self.as_polynomial = self.rat_ring(given_data)
      except TypeError:
        raise TypeError('Could not understand data given. Needs it successfully coerced to {}'.format(self.rat_ring))
    # If not, we try to convert either way.
    # We will leave this if-else division here, in case we eventually change the behavior of this method
    else:
      try:
        self.as_polynomial = self.rat_ring(given_data)
      except TypeError:
        raise TypeError('Could not understand data given. Need it successfully coerced to {}'.format(self.rat_ring))
    # Now we verify things.
    assert self.as_polynomial.degree() <= self.n_bound
    assert len(self.rat_ring.gens()) <= self.j_bound

  # Method __str__ 
  # Formats the instance to be printed in a human-friendly string
  # Let's just print self.as_polynomial. In the future we can add options for longer forms and whatever.
  def __str__(self):
    r'''
    Magic method. Returns a human-readable representation of the instance.
    '''
    return str(self.as_polynomial)

######################################################################

# Class GeneratorHomologyBU
# Not a subclass of ElementHomologyBU, but related
# One instance is an element of K_0(BU(n)) for a specific n
# Even if the element is in K_2d(BU(n)), we won't change the notation for it.
# There are 
class GeneratorHomologyBU(object):
  r'''
  Class of objects to represent a generator of (KU_p^hat)_0(BU) or of any (KU_p^hat)_2d(BU).
  
  It is represented by a vector, self.vector, which is a list that encodes a monomial in the beta_j's starting at j = 1.
  
  self.p is the prime where the K-theory is completed. It plays no role in the class.
  
  self.j_bound means the object belongs to the polynomial ring QQ(beta_0, beta_1, ..., beta_{self.j_bound}).
  In particular, len(self.vector) <= self.j_bound+1.
  
  self.n_bound refers to a maximum possible degree of the monomial, meaning the generator is in (KU_p^hat)_0(BU(n_bound)).
  The total degree is sum(self.vector), and we have self.total_degree() <= self.n_bound.
  
  EXAMPLE: to self.vector = [9, 0, 4, 8, 0] we associate the monomial beta_1^9*beta_3^4*beta_4^8
  
  EXAMPLE: to self.vector = [0, 0, ..., 0] we associate the monomial 1, also known as beta_0
  '''

  # Method __init__
  # Initializes the instance of the class
  def __init__(self, given_p, given_n_bound, given_j_bound, given_vector):
    r'''
    Magic method. Initializes an instance of the class.
    '''
    self.p = given_p
    # n_bound states that the total degree is <= n_bound, and therefore the generator belongs to BU(n_bound)
    # j_bound states that the highest beta_j involved is <= j_bound, and equivalently len(self.vector) <= j_bound
    self.n_bound = given_n_bound # Corrected only if inconsistent with total degree
    self.j_bound = given_j_bound # Corrected only if inconsistent with len(self.vector)
    # Why not already create a rat_ring
    # We will use a ring with coefficients in QQ as they are quicker and display better than Zp(self.p)
    self.rat_ring = PolynomialRing(QQ, self.j_bound, string_of_good_variables(1, self.j_bound+1))
    # The generators are also useful (otherwise it's hard to write them). Let's preface the list with a beta_0 or 1 in the same ring.
    self.augmented_gens = [self.rat_ring(1)]+list(self.rat_ring.gens())
    self.vector = given_vector
    # We will separate the formatting instructions
    self.verify_vector_consistency

  # Method verify_vector_consistency
  # Puts the vector in a specific format, and undoes possible inconsistencies
  def verify_vector_consistency(self):
    r'''
    Ensures all the attributes of self are consistent with each other.
    '''
    # By definition, every generator of K_0BU(n) has total degree <= n
    # If generator doesn't belong to K_BU(n), we adjust n to be total degree
    # If self.n is higher than total degree, it isn't a real problem. We could correct it with the deprecated clean_vector() if we really wanted
    if self.total_degree() > self.n_bound:
      self.n_bound = self.total_degree()
      print('Warning: This generator belongs to K_{0}BU({}) (and any higher). Fix: self.n_bound adjusted to exactly {}.'.format(self.n_bound, self.total_degree()))
    # Now we verify if the given number of variables, j_bound, is adequate and coherent with the vector
    # We check if self.vector doesn't 
    if len(self.vector) > self.j_bound:
      # Depending if these entrues
      if self.vector[self.j_bound:] != [0]*(len(self.vector) - self.j_bound):
        self.j_bound = len(self.vector)
        print('Warning: {} variables formally needed. Fix: self.vector unaltered and self.j_bound increased to exactly {}.'.format(self.j_bound, len(self.vector)))   
      else:
        self.j_bound = len(self.vector)
        print('Warning: {} variables really needed. Fix: self.vector unaltered and self.j_bound increased to exactly {}.'.format(self.j_bound, len(self.vector)))
    # Question: Should we make len(self.vector) longer to match self.j_bound?
    # It afraid this could save some errors in compute_adams_operation_of_monomial, even if increases the memory marginally
    # Nonetheless, we don't do it
    # To finalize, we verify the entries of the vectors are nonnegative integers
    for number in self.vector:
      assert check_if_number(number, 'ZZ>=0'), 'Every entry in self.vector should be a nonnegative integer.'
    return 'Warning: verify_vector_consistency() works as procedure, not as function. It modifies the instance. Don\'t return it like you just did.'

  # Method __hash__ produces a hash of the class instance
  # Python needs hashing to produce lists and dictionaries
  # To make it easier, simply rely on Python hash for tuples
  def __hash__(self):
    r'''
    Magic method. Returns a hash for the instance.
    '''
    return hash(tuple(self.vector.extend(self.p, self.n_bound, self.j_bound)))

  # Method __str__ formats the instance to be printed in a human-friendly string
  # In future allow for texification (or create another method for printing a texified text)
  # With optional arguments as "sparse = True" and "texify = True"
  def __str__(self):
    r'''
    Magic method. Returns a human-readable representation of the instance.
    '''
    if self.is_one():
      # Remember beta_0 = 1. We will display 1. It can be accomplished as below
      return_string = str(self.create_sage_monomial())
    else:
      return_string = str(self.create_sage_monomial())
    return return_string

  # Method __repr__ produces code that if interpreted generates the instance back
  # Probably not perfect, but should help the user a bit
  def __repr__(self):
    r'''
    Magic method. Returns an unambiguous representation of the instance.
    '''
    return_string = 'GeneratorHomologyBU(%d, %d, %d, %s)' % (self.p, self.n_bound, self.j_bound, self.vector)
    return return_string

  # Method create_sage_monomial
  # Creates a Sage monomial out of an element
  def create_sage_monomial(self):
    r'''
    Returns the monomial generated by the instance.
    '''
    # self.rat_ring.gens() is a tuple (beta_1, beta_2, ... beta_{j_bound}) (we don't use it anymore)
    # The native Sage method monomial() does all the hard work, as long as we do * on self.vector
    # We need to be aware that monomial only works for polynomials in multiple variables
    # Or not... not sure what is the deal with 1...
    instance_as_monomial = self.rat_ring.monomial(*self.vector)
    return instance_as_monomial

  # Method is_one
  # Determines if an instance if the 1, also known as beta_0
  # Good so we can write beta_0 if we want for it in __str__
  def is_one(self):
    r'''
    Returns whether the instance corresponds to beta_0 or u^d*beta_0.
    '''
    if self.total_degree() == 0:
      return True
    else:
      return False

  # Method total_degree
  # Produces the total degree of that generator
  def total_degree(self):
    r'''
    Returns the degree of the generator associated to the instance.
    '''
    sum_of_degrees = sum(self.vector)
    return sum_of_degrees

  # Method produce_place
  # Produces the place of a generator, given a maximum j
  # If not maximum k is given, it would be impossible as it isn't a well-order
  # For example, \beta_{20000} would before b_{1}^{2} with this evaluation
  # With max_j specified, there are Q(n, max_j) = C(n+max_j-1, max_j-1) generators with total degree (evaluation) n
  # The formula below produces the order (the place, the ordinal) of the generator in the well-order
  def produce_place(self, max_j):
    r'''
    Returns the index the monomial would assume in a degrevlex ranking in max_j variables.
    '''
    return get_degrevlex_ranking(self.vector, max_j)

  # Method compute_adams_operation_of_monomial
  # Given a generator, produces a list of coefficients (or maybe the generators themselves)
  # Also, remember beta_0 appear first on the lookup_table, and after that only concerns beta_1 through beta_(max_j)
  # Despite not being always needed, we always assume the lookup_table starts with beta_0
  def compute_adams_operation_of_monomial(self, lookup_table):
    r'''
    Returns an instance of ElementHomologyBU which is the image of the Adams operation on the instance.
    '''
    # Note r and d don't need to be passed as argument, as it use the lookup_table (which was calculated usind r and d)
    # (Length of self.vector)+1, and both dimensions of lookup_table should be the same. (In the right context, both are max_j + 1)
    # (Technically lookup_table could be larger and it would still work, but let's do like this, let's impose this rigidity). For debugging:
    supposed_table_size = len(self.vector) + 1
    assert len(lookup_table) == supposed_table_size, 'Look-up table number of rows should be one more than the length of self.vector.'
    for row in lookup_table:
      assert len(row) == supposed_table_size, 'Look-up table number of columns should be one more than the length of self.vector.'
    # Recall the row with index j is exactly the image of beta_j (the coefficients B_jl with respect the other beta_l's)
    # We need a separate approach for beta_0. The method self.is_one() detects it
    if self.is_one():
      # In this case, very easy
      image_as_polynomial = lookup_table[0][0]
    else:
      # Now the general case, where self.vector is not [0, 0, ..., 0]].
      # For each beta_j in the monomial, we need to find its image, and then multiply all self.total_degree() of them
      # It shouldn't be hard with self.rat_ring
      # To get psi^r of a monomial, we need to take the product of psi^r of each variable
      # Product starts as 1, and will have total_degree factors
      image_as_polynomial = self.rat_ring(1)
      # We need to do it for each (mathematical) variable (controlled by count)
      # The first count is what is referred in the thesis as j, and second_count is commonly denoted l. We will assume j and l are nonzero below
      for count in range(0, self.j_bound):
        # A number of times equal to its exponent in the expression, controlled by self.vector[count]
        # Note count is j-1 >= 0 and second_count is l >= 1. We shift those by start_at to find the corresponding rows and columns in lookup_table
        factor = (sum([lookup_table[count+1][second_count]*self.augmented_gens[second_count] for second_count in range(1, self.j_bound+1)]))**self.vector[count]
        image_as_polynomial *= factor
    # Product will be element of ElementHomologyBU.
    # We could output in other ways, but let's "factor into" ElementHomologyBU to make it easier
    return ElementHomologyBU(self.p, self.n_bound, self.j_bound, image_as_polynomial)

# Class FinitelyGeneratedAbelianGroup
# Used to make operations on group such that get order, direct sum
# Probably better to operate in other constructs as lists, strings or even native SageMath classes
# The native SageMath class for this doesn't accept strings as generators, as we want to do here
# We will try to construct an object such that:
# i) corresponds to a f. g. abelian group
# ii) comes from a quotient of modules (more precisely a cokernel)
# iii) has explicit generators which are equivalent classes of specific guys from an original group
# iv) that being the linear combination of generators/monomials beta_{j_1}...beta_{j_l} from K_0(BU) (or K_{2d}BU)
class FinitelyGeneratedAbelianGroup(object):
  r'''
  Class to represent finitely generated abelian groups as direct sum of cyclic groups defined by their order and a generator.
  '''

  # Method __init__
  # Responsible for creating an instance of this class
  # Pondering: should we add the self.base_ring? In the end, we kept simply abelian group, no p-adic nor p-local module structure.
  # group_data is either:
  # i) a list of summands (each summand a tuple (order, generator)), as a one-item tuple
  # ii) a list of orders then a list of generators (currently strings), as a two-item tuple
  def __init__(self, name, *group_data):
    r'''
    Magic method. Initializes an instance of the class.
    '''
    # We need to use the case len(group_data) == 0 for [] which isn't reobtainable from the technique *[]...
    # This would be the "zero group", the group with one element
    if len(group_data) == 0:
      self.list_of_summands = []
      self.list_of_orders = [summand[0] for summand in self.list_of_summands] # i.e. empty list
      self.list_of_generators = [summand[1] for summand in self.list_of_summands] # i.e. empty list
    # In this case, we have been given a list of summands, each summand a pair (order, generator)
    if len(group_data) == 1:
      self.list_of_summands = group_data[0]
      self.list_of_orders = [summand[0] for summand in self.list_of_summands]
      self.list_of_generators = [summand[1] for summand in self.list_of_summands]
    # In this case, we have been given two lists of same length, the first with orders, the second with generators
    elif len(group_data) == 2:
      self.list_of_orders = group_data[0]
      self.list_of_generators = group_data[1]
      # We could get None as generators. In this case, we prepare a ist of empty strings.
      # They can are renamed to gen_count below (doing separately we catch the case of len(group_data) == 1)
      assert len(self.list_of_orders) == len(self.list_of_generators), 'self.list_of_orders and self.list_of_generators should have same length'
      # For Python2, zip() provides a list. For Python 3, we need to do list(zip()) instead.
      self.list_of_summands = list(zip(self.list_of_orders, self.list_of_generators))
    else:
      raise TypeError('There should be exactly one or exactly two lists as group_data')
    assert isinstance(self.list_of_summands, list), 'self.list_of_summands should be a list'
    assert isinstance(self.list_of_orders, list), 'self.list_of_orders should be a list'
    assert isinstance(self.list_of_generators, list), 'self.list_of_generators should be a list'
    for count in range(len(self.list_of_orders)):
      assert check_if_number(self.list_of_orders[count], ['ZZ'], accept_infinity = True), 'self.list_of_orders[count] should be int or Sage integer or PlusInfinity'
      assert isinstance(self.list_of_generators[count], SAFE_BASESTRING), 'self.list_of_generators should be a str or basestring'
      assert isinstance(self.list_of_summands[count], tuple), 'self.list_of_summands[count] should be a tuple'
      assert check_if_number(self.list_of_summands[count][0], ['ZZ'], accept_infinity = True), 'list_of_summands[count][0] should be int or Sage integer or PlusInfinity'
      assert isinstance(self.list_of_summands[count][1], SAFE_BASESTRING), 'self.list_of_summands[count][1] should be a str or basestring'
    # We need to consider that generators are presented and when they aren't...
    # Let's say that that if we don't want to pass generators, we pass an empty list
    # (option of passing empty list may or may not be deprecated... we'll keep the code regardless)
    # And we then default to generic names if they are not present, or if they are empty strings
    # Note: now, we only alter it if empty strings are passed
    # If the sequence is empty, it will fail the assertion len(self.list_of_orders) == len(self.list_of_generators)
    for count in range(len(self.list_of_generators)):
      if self.list_of_generators[count] in ['', ' ']:
        self.list_of_generators[count] = ['gen_{}'.format(count+1)]
    # Currently we don't force it to be a (Z_p^hat)-module... we accept any ring... and treat is as abelian group
    # This group should be a module over self.base_ring... if anything a ZZ-module
    # Name is a string to attach a name to the group... maybe explain its origins, for example
    assert isinstance(name, SAFE_BASESTRING), 'name should be a str or basestring'
    if name: # i.e. not empty
      self.name = name
    else:
      self.name = 'A finitely generated abelian group without a name'

  # Method __repr__
  # Returns a string which, if run, initializes an instance equivalent to self
  def __repr__(self):
    r'''
    Magic method. Returns an unambiguous representation of the instance.
    '''
    return_string = 'FinitelyGeneratedAbelianGroup({}, {})'.format(self.name, self.list_of_summands)
    return return_string

  # Method __str__
  # Returns how we represent an instance as a string if mandated by a print function
  # Since we have many ways of printing (in another function), this will call our favorite one
  def __str__(self):
    r'''
    Magic method. Returns a human-readable representation of the instance.
    '''
    return self.print_flexibly('full')

  # Method print_flexibly
  # Workings mostly copied from (currently not deleted) functions list_of_summands_as_list_of_strings
  # and list_of_summands_as_single_string
  # But we could start with the name in the first line, depending on the option
  def print_flexibly(self, printing_choice = 'full'):
    r'''
    Returns expressions of self. Allows for many options of output.
    
    INPUT: One among 'full', 'name_only', 'orders_only', 'summands_only', 'name_and_orders', 'single_line'
    '''
    if printing_choice == 'name_only':
      return self.name    
    else:
      orders_only_string = (' '+DIRECT_SUM_SYMBOL+' ').join([get_cyclic_group_of_given_order(order) for order in self.list_of_orders])
      # If we want only the orders
      if printing_choice == 'orders_only':
        return orders_only_string
      elif printing_choice == 'name_and_orders':
        return self.name+'\n'+orders_only_string
      # If we want name and orders
      # Now we prepare multiple strings, with a trigger for one-element groups so they are treated differently
      elif not self.list_of_summands:
        list_of_summands_as_strings = ['One-element group']
      else:
        list_of_summands_as_strings = [print_summand(*summand) for summand in self.list_of_summands]
      # Now we serve them according to what is asked
      if printing_choice == 'single_line':
        summands_string = (' '+DIRECT_SUM_SYMBOL+' ').join(list_of_summands_as_strings)
        return summands_string
      else:
        summands_string = ('\n'+DIRECT_SUM_SYMBOL+' ').join(list_of_summands_as_strings)
        # This triggers problems in Python 2 and 3
        # Something that may be tried on SageMathCell or a console:
        # a = '\u2295'.encode('utf-8')
        # print(a, type(a), repr(a), sep = ', ')
        # sa = str(a)
        # print(sa, type(sa), repr(sa), sep = ', ')
        # b = 'example \u2295'.encode('utf-8')
        # print(b, type(b), repr(b), sep = ', ')
        # sb = str(b)
        # print(sb, type(sb), repr(sb), sep = ', ')
        # The commented code works with Python 3 but the second half fails with Python 2
        # Issue only solved with variable DIRECT_SUM_SYMBOL
        if printing_choice == 'full':
          return self.name+':\n'+summands_string
        elif printing_choice == 'summands_only':
          return summands_string
        else:
          raise NameError('Only valid choices for printing_choice are \'full\', \'orders_only\', \'name_only\', \'summands_only\', \'single_line\'')

  # Method total_order
  # Returns the total_order, or simply order, of the group
  # Returns the number of elements, finite or infinite
  # We call it total_order as order is a word used in other contexts too within the code
  # Nonetheless, in abelian group theory, order is the perfect word describing what this produces
  def total_order(self):
    r'''
    Returns the order of the abelian group, possibly infinite.
    '''
    # Note that in Sage we have oo == +Infinity, and even id(oo) == id(+Infinity)
    # Also, for this method, we need to have a special provision for order of a summand being 0
    if (0 in self.list_of_orders) or (+Infinity in self.list_of_orders):
      return +Infinity
    else:
      # Works with empty list just fine, outputting 1
      return special_product(self.list_of_orders)

# Class NonDecreasingIntegerList
# Like a list, but has a special comparation method suitable for our questions
class NonDecreasingIntegerList(list):
  r'''
  Class for nondecreasing lists of positive integers or +Infinity, to be ordered in a special sense consistent with monomorphisms in group theory.
  
  SEE ALSO: __lt__()
  '''

  # Method __init__
  # Overwrites the __init__ from the class list, but calls it in the middle
  # Besides that, checks if list is indeed nondecreasing and of integers
  def __init__(self, *args, **kwargs):
    r'''
    Magic method. Initiates an instance of the class.
    '''
    # Basically we do the same as we would do to a normal list
    list.__init__(self, *args, **kwargs)
    # Positive integer only
    assert all(check_if_number(item, ['ZZ>=0'], accept_infinity = True) for item in self), 'each item of self should be a positive integer or +Infinity'
    # We could sort item here with sorted(), but instead we require it to start sorted as nondecreasing
    assert self == sorted(self)

  # Method less_than_by_specific_increase
  def less_than_by_specific_difference(self, other_list, increase, difference_between_lengths):
    r'''
    Returns whether another instance of the class can be obtained from self
    via a specific number of insertions of 1's to the start of the list
    and increases by 1's to existing numbers in the list.
    
    SEE ALSO: __lt__()
    '''
    list_of_pairs_of_partitions = ''
    for going_to_be_appended_as_new_items in range(0, increase+1):
      going_to_be_added_to_existing_items = increase - going_to_be_appended_as_new_items
      for partition_of_new in Partitions(going_to_be_appended_as_new_items, length = difference_between_lengths).list():
        # We would normally write self_with_added_items = sorted(self+partition_of_new)
        # But since partiton_new is not of type list but of sage.combinat.partition.Partitions_nk_with_category.element_class
        # So we decide to do the appending in a manual way
        self_with_added_items = self[:]
        for appending_number in partition_of_new:
          # We allow the numbers to be 0 too
          # In our problem, this won't happen, but here we allow
          self_with_added_items.append(appending_number)
        self_with_added_items = sorted(self_with_added_items)
        # Now the two lists have the same length, and are both monotone
        # If all items of the self_with_added_items are less than or equal to the corresponding item in other_list
        #then it's easy to see that it's possible to add positive integers to the items of self_with_added_items
        #to make it equal to other_list (just add the difference, which is always zero or positive!)
        if all(self_with_added_items[count] <= other_list[count] for count in range(len(other_list))):
          return True
    return False   

  # Method __lt__
  # Characterized by insertions/containment and adding integers to elements
  # A list is less than the other_list if there is a partition of the difference between the sums (of each list)
  #which, if appended to self, would be equal to the other_list
  def __lt__(self, other_list):
    r'''
    Magic method. Describes the result of the operator <.
    
    The order relation < in the instances of this class is generated by:
    
    [a_1, ..., a_i, ..., a_n] < [a_1, ..., a_i +1, ..., a_n]
    
    and
    
    [a_1, ..., a_n] < [1, a_1, ..., a_n]
    '''
    assert isinstance(other_list, NonDecreasingIntegerList), 'other_list should be a NonDecreasingIntegerList'
    # Given that we check the last one, we must first think what happens if self or other_list are empty
    # Seems convoluted, but we don't want to do try/except. We will do embedded if's.
    # Empty list is less than any other (except itself)
    if self:
      if other_list:
        pass
      else:
        return False
    else:
      if other_list:
        return True
      else:
        # Two empty lists are equal to each other, __lt__ is False
        return False
    # First we deal with the case that any of them has an infinite value, necessarily a +oo at the end
    # If only one ends at +oo, that one is the largest (so having infinite value tops length in determining __lt__)
    if self[-1] == +oo and other_list[-1] != +oo:
      return False
    elif self[-1] != +oo and other_list[-1] == +oo:
      return True
    elif self[-1] == +oo and other_list[-1] == +oo:
      # If both end at infinity, we drop the +Infinity at the end of both, creating a (finite) recursion
      new_self = NonDecreasingIntegerList(self[:-1])
      new_other_list = NonDecreasingIntegerList(other_list[:-1])
      return new_self.__lt__(new_other_list)
    else:
      # Ok, only finite values.
      # We will do many tests. First we check the sum and length
      difference_between_sums = sum(other_list) - sum(self)
      difference_between_lengths = len(other_list) - len(self)
      if difference_between_sums <= 0 or difference_between_lengths < 0:
        return False
      else:
        # Now the main thing, which is in a separate function for modularity
        return self.less_than_by_specific_difference(other_list, difference_between_sums, difference_between_lengths)

  # Method __gt__
  # Defined in terms of other comparisons
  def __gt__(self, other_list):
    r'''
    Magic method. Describes the result of the operator >.
    '''
    return other_list.__lt__(self)

  # Method __le__
  # Defined in terms of other comparisons
  def __le__(self, other_list):
    r'''
    Magic method. Describes the result of the operator <=.
    '''
    return self.__lt__(other_list) or self.__eq__(other_list)

  # Method __ge__
  # Defined in terms of other comparisons    
  def __ge__(self, other_list):
    r'''
    Magic method. Describes the result of the operator >=.
    '''
    return self.__gt__(other_list) or self.__eq__(other_list)

######################################################################
# CODE: MISCELLANEOS UTILITARY FUNCTIONS
######################################################################

# Function get_ring_of_coefficients_from_k
# To get Z_p_hat or Z/p^kZ
def get_ring_of_coefficients_from_k(p, k):
  r'''
  Returns, after a proper format, Z_p^hat if k == 0 and Z/(p^k)Z if k > 0.
  '''
  if k == 0:
    return 'Z_{}^hat'.format(p)
  else:
    return 'Z/({}^{})Z'.format(p, k)

# Function get_cyclic_group_of_given_order
# Was a method for some time, which may or may not be the case in the future
# Z/nZ if n>0 and ZZ if n=0 or n=+Infinity (which is exactly the same as oo in Sage)
# Returns are strings... not worth creating a class just to denote cyclic groups
def get_cyclic_group_of_given_order(order):
  r'''
  Returns the cyclic group of a given order, finite or infinite.
  '''
  assert check_if_number(order, ['ZZ'], accept_infinity = True)
  if order == oo or order == 0:
    cyclic_group = 'Z'
  else:
    cyclic_group = 'Z/{}Z'.format(order)
  return cyclic_group

# Function quotient_ring_by_element
# Trying to figure out how to merge quotient_ring_by_element() and get_cyclic_group_of_given_order()
# Function quotient_ring_by_element
# Given an order, finite or infinite, we produce R or R/orderR.
def quotient_ring_by_element(order, ring = 'ZZ'):
  r'''
  Returns a string representing quotient ring of a ring by the ideal generated by an element of specific order.
  '''
  # Note that oo is +Infinity in Sage. We can say Z has order +Infinity (in comparison with Z/numberZ)
  # Sometimes the order comes as 0 for an infinite cyclic group. We make sure we catch it too.
  # To expand the functionality in the future, we should define canonical representations of different rings such as Z_p^hat and others.
  if str(ring) in ['Z', 'ZZ', 'Integer Ring']:
    ring_after_quotient = get_cyclic_group_of_given_order(order)
  else:
    # Situation is not as nice as for ZZ. Nonetheless, we try our best
    if isinstance(ring, SAFE_BASESTRING):
      # If given as string already, we don't do parenthesis
      # This can certainly backfire. It is a risk we take
      ring_after_quotient = '{}/{}{}'.format(ring, order, ring)
    else:
      # In this case format makes it into a string, and we add parenthesis to be extra sure
      ring_after_quotient = '({})/{}({})'.format(ring, order, ring)
  return ring_after_quotient

# Function print_summand
# First argument should be the order (0 meaning infinity, or a positive integer), second argument a generator (currently strings only)
def print_summand(order, generator):
  r'''
  Return a string representation of a summand in a cyclic group with a given generator.
  '''
  assert check_if_number(order, ['ZZ>=0'], accept_infinity = True), 'order should be an integer or infinity'
  assert isinstance(generator, SAFE_BASESTRING), 'generator should be a str or basestring'
  returning_string = '({}) < {} >'  .format(get_cyclic_group_of_given_order(order), generator)
  return returning_string

# Function list_of_summands_as_list_of_strings
# Allows us to design an uniform way to present a cokernel as a string.
# Should take a list of summands of the cokernel, and output a list of strings.
# Used in produce_cokernel, cokernel_of_adams_operation, maybe others, when they are asked to output a single string
# This allows for producing a string vertically (one direct summand per line) and also in a single line
# May be deprecated in the future for 
def list_of_summands_as_list_of_strings(list_of_summands, add_trivial_group_if_empty = False):
  r'''
  Returns a list of strings when given a list of summands.
  
  SEE ALSO: print_summand()
  '''
  # The argument cokernel should be a list with cokernel summands
  # Each cokernel summand is a list with two elements: in position 0 the order of the group, and in position 1 its generator
  # This could be done via dictionaries and tuples too. But we'll do with a list of lenght 2.
  list_of_strings = ['({}) < {} >'.format(cyclic_group_of_given_order(summand[0]), summand[1]) for summand in list_of_summands]
  # If there are no groups in the direct sum, we have an option of adding an object to the list, '1'
  if add_trivial_group_if_empty == True:
    if list_of_strings == []:
      list_of_strings = ['1']
  return list_of_strings

# Function list_of_strings_from_summands_as_single_string
# Displays a group given by summands in a single, multi-line string
def list_of_summands_as_single_string(list_of_summands, display_orientation = 'horizontal', add_trivial_group_if_empty = False):
  r'''
  Returns an abelian group or a list of summands formatted to fit into a single string.
  '''
  list_of_strings = list_of_summands_as_list_of_strings(list_of_summands, add_trivial_group_if_empty = add_trivial_group_if_empty)
  if display_orientation[0].lower() == 'h':
    # We can obtain a string in unicode by starting it with u. In this case, \u2295 is the direct sum symbol.
    return_string = (' '+DIRECT_SUM_SYMBOL+' ').join(list_of_strings)
  elif display_orientation[0].lower() == 'v':
    return_string = '\n'.join(list_of_strings)
  else:
    print('Warning. Not a valid choice of display orientation. Defaulted to displaying horizontally.\n')
    return_string = (' '+DIRECT_SUM_SYMBOL+' ').join(list_of_strings)
  return return_string

# Function right_st_nd_rd_th
# Finds the right suffix
def right_st_nd_rd_th(number):
  r'''
  Returns a string with the correct suffix for the ordinal form of the given number.
  '''
  assert check_if_number(number, 'ZZ>=0'), 'number should be a nonnegative integer'
  if number == 1:
    return '{}st'.format(number)
  elif number == 2:
    return '{}nd'.format(number)
  elif number == 3:
    return '{}rd'.format(number)
  else:
    return '{}th'.format(number)

# Function neat_u_power
# Print neatly powers of the Bott element u in KU_2
# Makes u^0 disappear from the notation, and add parenthesis for negative numbers
def neat_u_power(exponent):
  r'''
  Returns a proper string for when we multiply a homology class by the Bott element of K-theory.
  '''
  assert check_if_number(exponent, 'ZZ')
  if exponent == 0:
    return ''
  elif exponent > 0:
    return 'u^{}*'.format(exponent)
  else:
    return 'u^({})*'.format(exponent)

# Function string_of_good_variables()
# Helps defining a Sage multivariate polynomial ring
# Essentially, provides with a string to feed into PolynomialRing to name the variables, often starting at 0 or 1
# And also we can change them according to SYMBOL_FOR_GENERATORS
# We will follow the range convention of stopping just short of the second argument
def string_of_good_variables(start_at, one_after_the_last):
  r'''
  Returns string with specifically chosen variable names to form a polynomial ring in SageMath.
  '''
  assert check_if_number(start_at, 'ZZ'), 'start_at should be an integer'
  assert check_if_number(one_after_the_last, 'ZZ'), 'one_after_the_last should be an integer'
  assert start_at <= one_after_the_last, 'start_at should be less than or equal to one_after_the_last'
  return_string = ', '.join((SYMBOL_FOR_GENERATORS+'{}').format(count) for count in range(start_at, one_after_the_last))
  return return_string

# Function get_linebreaks_and_dashes
# To get a standard line of dashes, after a few line breaks
def get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes):
  r'''
  Returns a string with a given number of linebreaks '\n' and dashes '-'.
  '''
  string_with_linebreaks = '\n'*number_of_linebreaks
  string_with_dashes = '-'*number_of_dashes
  return string_with_linebreaks + string_with_dashes

# Function format_output_as
# Used for admitting a broader range of output options
# Used to determine how object results should be output by another function
# (We are writing this separately to be used by any function that needs this feature)
# Basically, considering four options for now:
# i) string, which generates a decent string for printing to screen (or even recorded, if the user wants)
# ii) csv, producing data in a way that can be usefully written to a CSV file
# iii) object, in which the function tries to return its output in the purest form possible (like a Sage construct or object)
# iv) latex, a LaTeX-friendlier output
# v) list, when providing data simply by numbers that codify the whole object (ex: coefficients instead of polynomial)
# vi) Last added, bool
# vii-xi) Added in different moments: txt, generator, single_line, dict, explain
def format_output_as(given_option):
  r'''
  Returns a shorter word for option of output from the user.
  '''
  # First we prepare it to be alphabetic only, lowercase and remove plural
  lower_string = ''.join([character for character in given_option if character.isalpha()])
  lower_string = lower_string.lower()
  if lower_string[-1] == 's':
    lower_string = lower_string[0:-1]
  # Now we start checking the possibilities.
  if lower_string in ['tex', 'latex', 'pdf']:
    formatted_string = 'tex'
  elif lower_string in ['csv', 'excel', 'spreadsheet']:
    formatted_string = 'csv'
  elif lower_string in ['file', 'txt']:
    formatted_string = 'txt'
  elif lower_string in ['bool', 'boolean']:
    formatted_string = 'bool'
  elif lower_string in ['dict', 'dictionary']:
    formatted_string = 'dict'
  elif lower_string in ['string', 'text', 'str', 'screen', 'user', 'friendly']:
    formatted_string = 'string'
  elif lower_string in ['object', 'obj', 'pure', 'sage', 'sagemath', 'construct', 'poly', 'polynomial', 'matrix', 'vector']:
    formatted_string = 'object'
  elif lower_string in ['generator', 'gen']:
    formatted_string = 'generator'
  elif lower_string in ['list', 'table', 'tuple']:
    formatted_string = 'list'
  elif lower_string[0:5] == 'single':
    formatted_string = 'single_line'
  elif lower_string[0:2] == 'exp' or lower_string[0:3] == 'long' or 'detail' in lower_string:
    formatted_string = 'explain'
  else:
    print('Error reading output format option. Defaulted to \'object\'.')
    print('Valid options: tex, dict, csv, txt, bool, string, object, generator, list, single_line.\n')
    formatted_string = 'object'
  return formatted_string

# Function format_what_to_do
# Like format_output_as, to tentatively simplify things and be more flexible
# Still has plenty of room for improvement
def format_what_to_do(given_option):
  r'''
  Returns a valid two-character option of action streamlining the user's input.
  '''
  if len(given_option) == 2:
    return POSSIBLE_ACTIONS[given_option.lower()]
  else:
    if given_option.lower() in POSSIBLE_ACTIONS.values():
      return given_option
    else:
      raise ValueError('Only valid what_to_do options are the ones specified in the dict POSSIBLE_ACTIONS')

# Function format_direct_or_pre
# For the method find_pre_image_of_vector, tell if we want the pre-image or the direct image
def format_direct_or_pre(given_option):
  r'''
  Returns a shorter word for the option of user regarding taking a pre-image or a direct image.
  '''
  # First we catch the cases of inverse image and pre-image
  if given_option[0:3].lower() in ['pre', 'inv']:
    formatted_option = 'pre'
  # Now we do the cases of direct image, or forward image
  else:
    formatted_option = 'direct'
  return formatted_option

# Function check_monotonicity
# Checks all 4 kinds of monotonicity in a list (or any iterable, methinks)
# Has option to ignore or include infinite values
def check_monotonicity(list_of_comparables, monotonicity_type, include_infinite_values = False):
  r'''
  Returns whether a list satisfies a given type of monotonicity.
  '''
  # Note that list_of_comparables can be a list of anything. We can compare numbers to numbers, but also lists to lists
  # And this is (mainly) to be used to show that a list of lists or a list of numbers is monotone nondecreasing
  if not include_infinite_values:
    # If it's a list of numbers, them we exclude the infinite values.
    # If it's a list of lists (or other kind of comparables), we don't do this
    if all(check_if_number(item, ['RR'], True) for item in list_of_comparables):
      list_of_comparables = [item for item in list_of_comparables if not RR(item).is_infinity()]
  if monotonicity_type == '<=':
    is_monotone = all(x <= y for x, y in zip(list_of_comparables, list_of_comparables[1:]))
  elif monotonicity_type == '<':
    is_monotone = all(x < y for x, y in zip(list_of_comparables, list_of_comparables[1:]))
  elif monotonicity_type == '>=':
    is_monotone = all(x >= y for x, y in zip(list_of_comparables, list_of_comparables[1:]))
  elif monotonicity_type == '>':
    is_monotone = all(x > y for x, y in zip(list_of_comparables, list_of_comparables[1:]))
  else:
    raise ValueError('monotonicity type should be \'<=\', \'<\', \'>=\' or \'>\'.')
  return is_monotone

# Function force_check_old_way_cokernel
# Checks if passed argument is a cokernel (obtained our ways), returning it the old way
# It could be either an instance of FinitelyGeneratedAbelianGroup
# Or the old way, a list of tuples like [(order_1, generator_1), ... , (order_whatever, generator_whatever)]
# If none of this, then raise ValueError
# If check_only == True, then return True or False only (if thing is cokernel or not). It checks but doesn't force
def force_check_old_way_cokernel(potential_cokernel, check_only = False):
  r'''
  Returns a finitely generated abelian group as a list of pairs (order, generator).
  
  Can also be used to tell whether the given data is already in the format.
  '''
  # First we see if it's an instance of the class
  if isinstance(potential_cokernel, FinitelyGeneratedAbelianGroup):
    potential_cokernel = potential_cokernel.list_of_summands
  # Now we check if everything is all right
  # It may be redundant if it was from FinitelyGeneratedAbelianGroup, but it isn't
  # Since this is the function called to verify that the instance is all right
  if isinstance(potential_cokernel, list):
    if all(isinstance(summand, tuple) for summand in potential_cokernel):
      if all(len(summand) == 2 for summand in potential_cokernel):
        if all(check_if_number(summand[0], ['ZZ'], accept_infinity = True) and summand[0] >= 0 and isinstance(summand[1], SAFE_BASESTRING) for summand in potential_cokernel):
          if check_only == True:
            return True
          else:
            return potential_cokernel
        else:
          if check_only == True:
            return False
          else:  
            raise ValueError('Not a cokernel, each tuple in the list should be a pair (integer, string)')          
      else:
        if check_only == True:
          return False
        else:  
          raise ValueError('Not a cokernel, list should be of tuples of length 2')
    else:
      if check_only == True:
        return False
      else:  
        raise ValueError('Not a cokernel, list should be of tuples')
  else:
    if check_only == True:
      return False
    else:  
      raise ValueError('Not a cokernel, should be a list')

# Function check_if_number
# Checks if number is integer, rational, real
# Can check >0, >= 0 for ZZ (don't think we will need for the others), even if 'ZZ>0', 'ZZ>=0' are not rings
# In the future, maybe allow checking p-adicity-safe
def check_if_number(number, list_of_rings = ['RR'], accept_infinity = False):
  r'''
  Returns whether a number is a element of a given ring or a given list of rings.
  
  Has options for allowing infinite values, as well as restricting to positive or nonnegative elements only.
  '''
  # If a single ring is passed as string, we want
  if isinstance(list_of_rings, SAFE_BASESTRING):
    list_of_rings = [list_of_rings]
  # If list of rings is empty, we test if number is real (we use no complex numbers here)
  if not list_of_rings:
    list_of_rings = ['RR']
  # It has to be in at least one of the rings to be considered acceptable
  # It will be not acceptable until proved acceptable
  # We will accept Infinity at this moment, on the exceptions, but reject later. Note oo belongs to RR but not to ZZ nor QQ
  is_number_acceptable = False
  if ('Z' in list_of_rings) or ('ZZ' in list_of_rings):
    try:
      zz_number = ZZ(number)
      is_number_acceptable = True
    except TypeError:
      if number == oo or number == -oo:
        is_number_acceptable = True        
  if ('Z>0' in list_of_rings) or ('ZZ>0' in list_of_rings):
    try:
      zz_number = ZZ(number)
      if zz_number > 0:
        is_number_acceptable = True
    except TypeError:
      if number == oo:
        is_number_acceptable = True    
  if ('Z>=0' in list_of_rings) or ('ZZ>=0' in list_of_rings):
    try:
      zz_number = ZZ(number)
      if zz_number >= 0:
        is_number_acceptable = True
    except TypeError:
      if number == oo:
        is_number_acceptable = True    
  if ('Q' in list_of_rings) or ('QQ' in list_of_rings):
    try:
      qq_number == QQ(number)
      is_number_acceptable = True
    except TypeError:
      if number == oo or number == -oo:
        is_number_acceptable = True    
  if ('R' in list_of_rings) or ('RR' in list_of_rings):
    try:
      rr_number = RR(number)
      is_number_acceptable = True
    except TypeError:
      pass
  # If we don't accept infinite and number is infinite, we flag it as non-acceptable
  if not accept_infinity:
    try:
      rr_number = RR(number)
      if rr_number.is_infinity():
        is_number_acceptable = False
    except TypeError:
      pass
  return is_number_acceptable

# Function get_range_from_number
# If a number is passed where we expect a list, we create a range (as a list)
# If already a list or iterable, we keep it, even if not of consecutive numbers
# Have options to different start, and to include odd primes only
# If tuple, force a list
# If start_at = None, create list of single element (same effect of setting start_at = given_start)
# If given number is less than start_at, go up from it, stopping at start_at
def get_range_from_number(given_arg, start_at = None, force_odd_prime = False):
  r'''
  Returns a range or interval for a given number.
  
  If `given_arg` is already an iterable, returns it as a list of increasing numbers.
  
  If `given_arg` is an integer, then return a range whose first and last elements are `given_arg` and `start_at`.
  In this case, if `start_at` is None, return the length 1 list [give_arg].
  
  We have an option to exclude numbers which are not positive odd primes from the result.
  '''
  # We first determine if we are getting an iterable, in which case we return it as a list
  # We use the method len() to detect them with a try/except pair
  try:
    variable_for_length = len(given_arg)
    # Let's check we have good items inside given_arg, and not a string (which also has a len() method)
    for item in given_arg:
      assert check_if_number(start_at, ['ZZ']), 'If given_arg has items each should be an integer'
    # Make it into a list via comprehension, in increasing order.
    # (The argument start_at is completely ignored)
    sorted_given_arg = sorted(given_arg)
    if force_odd_prime:
      resulting_arg = [number for number in sorted_given_arg if (number >= 3 and is_prime(number))]
    else:
      resulting_arg = [number for number in sorted_given_arg]
  except TypeError:
    # In this case we want given_arg to be single integer
    assert check_if_number(given_arg, ['ZZ']), 'If given_arg is not an iterable it should be an integer'
    assert check_if_number(start_at, ['ZZ']) or start_at == None, 'start_at should be an integer or None'
    if start_at == None:
      # In this case we want to form the list of one element, given_arg. We set start_at = given_arg
      start_at = given_arg
    # Now we figure out if we should do [start_at..given_arg] or [given_arg..start_at] (list always increasing)
    # It can be done in the two following lines
    lowest_number = min(start_at, given_arg)
    highest_number = max(start_at, given_arg)
    # We generate lists by comprehension
    if force_odd_prime:
      resulting_arg = [number for number in range(max(3, lowest_number), highest_number+1) if is_prime(number)]
    else:
      resulting_arg = [number for number in range(lowest_number, highest_number+1)]
  # We are ready to deliver the results obtained from either path of the try/except
  assert isinstance(resulting_arg, list), 'resulting_arg should be a list'
  return resulting_arg

# Function list_from_matrix
# Given a Sage matrix, returns a list of lists, those last ones being the rows
def list_from_matrix(the_matrix):
  r'''
  Returns a list of lists corresponding to a SageMath matrix.
  '''
  # Each Sage matrix can be iterated over the row, which are inherently Sage vectors.
  # We can make a list with eachrow vector, then group them in a list of those lists.
  matrix_as_list = [list(row) for row in the_matrix]
  return matrix_as_list

# Function safe_input
# Solves the problem of input for versions 2.x and 3.x of Python seamlessly
# description_string is what the coder tells the user about the input
# Right now unused
def safe_input(description_string):
  r'''
  Returns `raw_input()` or `input()` depending of Python version.
  
  Allows for a version-agnostic command for input.
  '''
  try:
    # If we are in Python 2.x, raw_input() will work fine, as it what we want.
    return raw_input(description_string)
  except NameError:
    # In Python 3.x, the above fails, so we should use input() instead.
    return input(description_string)

# Function safe_basestring
# To work with Python 2 and 3, since basestring does not exist in Python 3
# Also, unicode strings are instances of str in Python 3 but not in Python 2
# Specially good to use as isinstance(variable, safe_basestring())
# An alternative, also version-agnostic would be (), but this ignores bytes
def safe_basestring():
  r'''
  Returns a tuple of string classes of the Python version being run.
  '''
  try:
    return_tuple = (basestring)
    # If we are in Python 2.x, this is enough.
    return return_tuple
  except NameError:
    # In Python 3.x, there is no basestring in the namespace, so we opt for (str, byte)
    return (str, bytes)

# Function find_cpu_count
# Gets the number of physical CPU cores in the computer the code is running on
# Right now unused
def find_cpu_count():
  r'''
  Returns the number of physical CPU cores of the user's computer.
  '''
  # Alternative: import psutil, then return psutil.cpu_count(logical = False)
  # We however always opt for having the smallest possible namespace, and for things to be the most explicit possible
  # And so we do the following (allied to a previous import)
  return cpu_count_from_psutil(logical = False)

# Function orders_only_from_what_to_do
# Determines if orders of the cokernel are sufficient or if the generators should be considered
def orders_only_from_what_to_do(what_to_do):
  r'''
  Returns the correct `orders_only` option from given `what_to_do` option.
  '''
  # We first format the option if it wasn't already
  formatted_what_to_do = format_what_to_do(what_to_do)
  # Now we divide the cases. Apparently, it is one against all others.
  if formatted_what_to_do == 'pc':
    orders_only = False
  else:
    orders_only = True

# Function get_the_logs_maybe
# Obtains only the decomposition of the cokernel, ignoring group generators
# Works with the three (now four!) kinds of approaches to infinity
# Works for list and a dict (returns same type)
# output_as not implemented
@parallel(NUMBER_OF_CORES)
def get_the_logs_maybe(p, bunch_of_cokernels, replace_orders_by_logs = True, output_as = 'list', orders_only = False):
  r'''
  Given a list or dict of cokernels, produces a list for each cokernel.
  
  It can be either a list of the orders for the groups whose direct sum form that cokernel,
  or a list of the logarithms of those same orders.
  '''
  if isinstance(bunch_of_cokernels, list):
    # We want to accept cokernels as lists of pairs (order, generator)
    # We work with cokernels the old way. If we have a FinitelyGeneratedAbelianGroup, we transform it via force_check_old_way_cokernel
    list_of_cokernels = [force_check_old_way_cokernel(cokernel) for cokernel in bunch_of_cokernels]
    list_of_list_of_orders = [[order_of_group(group) for group in cokernel] for cokernel in list_of_cokernels]
    if replace_orders_by_logs == True:
      list_of_list_of_logs = [[special_log(order, p) for order in list_of_orders] for list_of_orders in list_of_list_of_orders]
      returning_object = list_of_list_of_logs
    else:
      returning_object = list_of_list_of_orders
  else:
    # This function works with only lists and dicts
    # We will simply adapt the code for dicts
    assert isinstance(bunch_of_cokernels, dict), 'bunch_of_cokernels should be a list or a dict'
    dict_of_cokernels = {key: force_check_old_way_cokernel(bunch_of_cokernels[key]) for key in bunch_of_cokernels}
    dict_of_list_of_orders = {key:[order_of_group(group) for group in dict_of_cokernels[key]] for key in dict_of_cokernels}
    if replace_orders_by_logs == True:
      dict_of_list_of_logs = {key:[special_log(order, p) for order in dict_of_list_of_orders[key]] for key in dict_of_list_of_orders}
      returning_object = dict_of_list_of_logs
    else:
      returning_object = dict_of_list_of_orders
  # Right now no implementation for output_as... great nonetheless, methinks!
  return returning_object

# Function order_of_group
# At this moment, we don't make assumptions on how the groups are fed into this function
# If we are fed only the order, we do nothing.
# If we are fed a group as a pair (order, generator), we conserve the order (0th position) and we drop the generator
# If we are fed a whole group as a FinitelyGeneratedAbelianGroup, we get its [total] order
# This is supposed to be compatible with orders_only True and False
def order_of_group(group):
  r'''
  Returns the order of an abelian group.
  
  INPUT:
  
  `group`: A group which can be fed into the function in any one of the following ways:
  
  i) an instance of `FinitelyGeneratedAbelianGroup`, when we use `total_order()`
  
  ii) a list of many summands, each as (order, generator), and we compute the product of those orders
  
  iii) a single tuple (order, generator)
  
  iv) a single number, which becomes the order
  
  v) a list of orders, the product being the total order
  
  OUTPUT:
  
  The order of the group.
  '''
  if check_if_number(group, ['ZZ'], accept_infinity = True):
    return group
  elif isinstance(group, FinitelyGeneratedAbelianGroup):
    return group.total_order()
  else:
    # First we eliminate the empty list/tuple
    if not group:
      return 1
    # Now we make a provision for a tuple (order, generator), taking advantage of Python's lazy evaluation
    elif isinstance(group, tuple) and len(group) == 2 and check_if_number(group[0], ['ZZ'], accept_infinity = True) and isinstance(group[1], str):
      return group[0]
    # Not we determine if we received a list of orders or of pairs (order, generator)
    # The following comprehension takes care of separating out the generators if needed
    guaranteed_list_of_orders = [summand[0] if len(summand) == 2 else summand for summand in group]
    product_of_orders = special_product(guaranteed_list_of_orders)
    return product_of_orders

# Function special_log
# Round a positive integer to the integer closer to its logarithm in a base
# For 0 or +Infinity, return +Infinity (because 0 often means ZZ which has infinite order)
# For a negative number, return an error (since this is only for orders of groups)
def special_log(number, base):
  r'''
  Returns the logarithm of a number in a base, to be used when such logarithm is an integer. If the number is 0 or +Infinity, returns +Infinity.
  '''
  if number == 0 or number == +Infinity:
    return +Infinity
  elif number > 0:
    # Use int to round to nearest integer and get an 'int'.
    # Use ZZ to get a sage.rings.integer.Integer
    return ZZ(int(log(number, base)))
  else:
    raise ValueError('Can only take special_log of number between 0 and +Infinity, inclusive')

# Function
# Gives the maximum number of a list but with some options to deal with infinity
def special_max(given_list, accept_infinity = False):
  r'''
  Returns the maximum value of a list of nonnegative numbers, with an option of ignoring infinite values.
  '''
  if not accept_infinity:
    prepared_list = [item for item in given_list if not RR(item).is_infinity()]
  else:
    prepared_list = given_list
  # Need to deal with empty lists. Set maximum to 0 in this case.
  if not prepared_list:
    return 0
  else:
    return max(prepared_list)

# Function special_sum
# Works like sum, but may be asked to ignore infinite values
def special_sum(given_list, accept_infinity = False):
  r'''
  Returns the sum of values of a list of nonnegative numbers, with an option of ignoring infinite values.
  '''
  # We eliminate infinite values if asked. If not, the sum will become +Infinity if there is at least one +Infinity
  if not accept_infinity:
    prepared_list = [item for item in given_list if not RR(item).is_infinity()]
  else:
    prepared_list = given_list
  # Need to deal with empty lists. Set sum to be 0 in this case.
  if not prepared_list:
    return 0
  else:
    return sum(prepared_list)

# Function special_product
# Works like product, but may be asked to ignore infinite values
def special_product(given_list, accept_infinity = False):
  r'''
  Returns the product of the values of a list of nonnegative numbers, with an option of ignoring infinite values.
  '''
  # If not eliminated and present, product will become Infinity if there is at least one Infinity
  if not accept_infinity:
    prepared_list = [item for item in given_list if not RR(item).is_infinity()]
  else:
    prepared_list = given_list
  the_product = 1
  for number in prepared_list:
    the_product *= number
  return the_product

@parallel(NUMBER_OF_CORES)
# Function find_primitive_root_of_power
# Takes a power of a prime and returns the lowest primitive root
# Fact: every prime p has a primitive root g mod p
# Fact: if g is primitive root mod p, then either g or p+g is primitive root mod p^2
# Fact: if g is primitive root mod p^2, then g is primitive root mod p^k por any k
def find_primitive_root_of_power(p, k):
  r'''
  Returns a primitive root of a power of a prime.
  '''
  # First we try to find primitive roots of the prime (to save processing)
  # For coercion, we can use either sage.rings.integer.Integer or ZZ
  p = ZZ(p)
  k = ZZ(k)
  assert is_prime(p), 'p should be prime'
  assert k > 0, 'k should be a positive integer'
  if p == 2: # Big exception to the theory, 2 is the oddest prime
    primitive_root_p = 1
  elif p >= 3:
    # We test every number from 1 to p-1 until we find a primitive root for p
    modulo_p = IntegerModRing(p)
    candidate_powers = [(p-1)/prime for prime in prime_divisors(p-1)]
    primitive_root_p_found = False
    candidate_number = 2
    while primitive_root_p_found == False:
      this_one_proved_not_be = False
      for candidate_power in candidate_powers:
        if modulo_p(candidate_number)**candidate_power == modulo_p(1):
          this_one_proved_not_be = True
      if this_one_proved_not_be == False:
        primitive_root_p_found = True
        primitive_root_p = candidate_number
      else:
        candidate_number += 1
  # Now we work in finding a primitive root for p^k
  # It should be either primitive_root_p or primitive_root_p + p
  if k == 1:
    primitive_root_p_k = primitive_root_p
  elif k >= 2:
    if p == 2:
      # Don't have any special method, let's do the most bruteforcy way...
      # Some powers of 2 have no primitive root. 8 and 16 have no primitive roots
      modulo_2_k = IntegerModRing(2**k)
      candidate_root_2_k_found = False
      primitive_root_p_k = None # Right now there might or might not be one
      candidate_number = 3
      while primitive_root_2_k_found == False and candidate_number < p**k:
        this_one_proved_not_be = False
        for candidate_power in candidate_powers:
          if modulo_2_k(candidate_number**candidate_power) == modulo_2_k(1):
            this_one_proved_not_be = True
        if this_one_proved_not_be == False:
          primitive_root_p_k_found = True
          primitive_root_p_k = candidate_number
        else:
          candidate_number += 1
    elif p >= 3:
      modulo_p_2 = IntegerModRing(p**2)
      # Remember that it is enough to find a primitive root for k=2 if k >= 2
      candidate_powers = [euler_phi(p**2)/prime for prime in prime_divisors(euler_phi(p**2))]
      #print('For number '+str(p**2))
      #print('totient is '+str(euler_phi(p**2)))
      #print('and candidate powers are '+str(candidate_powers)+'\n')
      primitive_root_p_proved_not_work = False # Assume for now
      for candidate_power in candidate_powers:
        if modulo_p_2(primitive_root_p**candidate_power) == modulo_p_2(1):
          primitive_root_p_proved_not_work = True
          #print('Primitive root '+str(primitive_root_p)+' for prime '+str(p)+' doesn\'t work because')
          #print(str(primitive_root_p)+'**'+str(candidate_power)+' is congruent to 1 mod '+str(p)+'**2.')
      if primitive_root_p_proved_not_work == True:
        primitive_root_p_k = primitive_root_p + p
      else:
        primitive_root_p_k = primitive_root_p
  # Assertion to tell if really primitive power. Resource-consuming, though
  assert len(set([IntegerModRing(primitive_root_p_k)**power for power in range(euler_phi(p**k))])) == euler_phi(p**k), '{} not a primitive root of {}^{}'.format(primitive_root_p_k, p, k)
  return primitive_root_p_k

# Function get_primitive_roots_of_squares
# Gets the list of primitive roots of the squares of odd primes from 3 to number
# Actually a list of pairs (prime, lowest primitive root of its square)
# If fed a list, excludes the nonprime numbers
def get_primitive_roots_of_squares(range_for_p, output_as = 'list'):
  r'''
  Returns the primitive roots of the squares of a list of primes.
  '''
  # First we want to get a list of primes if we are given a number (the maximum prime) or a list containing composite numbers
  range_for_p = get_range_from_number(range_for_p, start_at = 3, force_odd_prime = True)
  #list_of_squares = [prime**2 for prime in list_of_primes]
  normalized_list = [sage.parallel.decorate.normalize_input((prime, 2)) for prime in range_for_p]
  generator = find_primitive_root_of_power(normalized_list)
  if format_output_as(output_as) == 'string':
    return_string = ''
    for line in generator:
      return_string += 'Prime: {} Square: {} Primitive root: {}\n'.format(line[0][0][0], line[0][0][0]**2, line[1])
    return return_string
  elif format_output_as(output_as) == 'generator':
    return generator
  else: 
    return_list = []
    for line in generator:
      return_list.append((line[0][0][0], line[1]))
    return return_list

######################################################################
# CODE: COMBINATORIAL FUNCTIONS
######################################################################

# Function get_stars_and_bars_solution
# Computes the number of monomials/generators with degree n in j variables
# Q(n, j) = C(n+j-1, j-1), and we use a native SageMath function for the binomial number
# Also known as "stars and bars" problem (n is number of stars, to be separated in j groups by j-1 bars)
def get_stars_and_bars_solution(n, j):
  r'''
  Returns the binomial number (`n`+`j`-1)!/(`n`!(`j`-1)!).
  
  This is also the solution to the stars-and-bars problem. It is the number of ways we can split a line of `n` objects called stars into `j` groups by j-1 dividers, or bars.
  
  It is also the number of monomials of degree `n` in `j` variables.
  '''
  return binomial(n+j-1, j-1)

# Function get_degrevlex_ranking
# Returns degrevlex ranking of specific monomial
def get_degrevlex_ranking(vector, max_j):
  r'''
  Returns the degrevlex ranking of a monomial on a polynomial ring on `max_j` variables.
  
  The monomial is represented by a vector as done in the class `GeneratorHomologyBU`.
  
  The convention is that the monomial 1 is the 0th monomial in this ordering.
  
  Thus, the ranking of the monomial/vector is the same as the number of monomials/vectors placed before it.
  
  SEE ALSO: GeneratorHomologyBU
  '''
  # If vector is longer, we want to throw a warning that the last coordinates will be ignored.
  if len(vector) > max_j:
    if vector[max_j:] == [0]*(len(vector)-max_j):
      # Throw the warning even if those are 0. Later we can change the behavior.
      print('Warning: Vector too long. All entries from max_j on are 0\'s, and will be ignored.')
    else:
      # In this case, the last variables will be ignored. We raise an error
      raise CustomException('len(vector) > max_j, impossible to produce ordinal in get_degrevlex_ranking.')
  # To make it work, we need max_j as the length of the vector
  vector.extend([0]*(max_j - len(vector)))
  # Now we count the generators/monomials/vectors prior to the given vector
  ordinal = 0
  ordinal += get_stars_and_bars_solution(sum(vector)-1, max_j+1)
  for count in range(2, max_j+1):
    ordinal += get_stars_and_bars_solution(sum(vector[0:count]), count)
    ordinal -= get_stars_and_bars_solution(sum(vector[0:count-1]), count)
  return ordinal

# Function compute_max_ordinal
# This is derived from the combinatorial problem of separating fixed_degree objects with max_j-1 bars
# (Often called stars and bars)
# We add then these numbers for fixed_degree from 0 to max_degree
def compute_max_ordinal(fixed_degree, max_j):
  r'''
  Returns the degrevlex ranking in `max_j` variables of the last monomial of degree `fixed_degree`.
  
  Equivalently, returns the number of monomials in `max_j` variables and degree at most `fixed_degree`, minus one. 
  '''
  # We subtract 1 to account from the fact that we start at the 0th monomial, 1 or beta_0
  computed_max_ordinal = sum([binomial(each_degree + max_j - 1, each_degree) for each_degree in range(0, fixed_degree+1)]) - 1
  return computed_max_ordinal

# Function compute_min_ordinal
# This is more or less used in two functions, so we better write it once
def compute_min_ordinal(fixed_degree, max_j):
  r'''
  Returns the degrevlex ranking in `max_j` variables of the first monomial of degree `fixed_degree`.
  '''
  # It is one after the max_ordinal for fixed_degree-1
  return compute_max_ordinal(fixed_degree - 1, max_j) + 1

# Function compute_degrevlex_rankings_for_stratum
# Given fixed_degree (or an) and max_j, gives the lowest and highest degrevlex ranking
# The strata is defined by the degree, and degrevlex ranking is on monomials on max_j variables
def compute_degrevlex_rankings_for_strata(interval_of_degrees, max_j):
  r'''
  Returns the lowest and highest degrevlex ranking on a specific stratum or strata.
  '''
  # This function should work accepting an interval of degrees or a single degree
  # To guarantee we have a list instead of a number, we do
  interval_of_degrees = get_range_from_number(interval_of_degrees, start_at = None)
  # Now we perform the following operations which are good for lists
  min_degree = ZZ(min(list(interval_of_degrees)))
  max_degree = ZZ(max(list(interval_of_degrees)))
  min_ordinal = compute_min_ordinal(min_degree, max_j)
  max_ordinal = compute_max_ordinal(max_degree, max_j)
  return (min_ordinal, max_ordinal)

# Function produce_first_places
# Return list of first max_ordinal lists according to the ordering
# Ordering exists only because the number of variables, max_j, is specified
# Note that having max_ordinal = 0 produces 1 element, [0,0,...,0], which is beta_0
# For Sage one uses deepcopy() while in Python one uses copy.deepcopy() imported from copy
# Note this produces a list of max_ordinal - min_ordinal + 1 lists/ordinals/monomials
def produce_first_places(min_ordinal, max_ordinal, max_j, output_as = 'list'):
  r'''
  Produces a specific interval of monomials in the order degrevlex.
  '''
  dict_of_all_ordinals = {}
  list_of_all_ordinals = []
  # We will keep going until done. We will create them from 0 (have no convenient way to start at the middle), but only append the requested ones
  for count in range(0, max_ordinal+1):
    if count == 0:
      # We need to create the very first (count = 0) explicitly
      # [0, 0, ... 0] is done manually. The numbers correspond to beta_1, ..., beta_(max_j)
      current_ordinal = [0]*(max_j)
    else:
      # Here is essentially how to move from one guy to the next in the order, creating the next current_ordinal from the prior
      # Note that the variable count is not used at all; it is only a matter of succeeding.
      for sub_count in range(max_j):
        # First we find the first position of the vector which has a nonzero element (unless is the last one)
        if sub_count < max_j - 1:
          if current_ordinal[sub_count] != 0:
          # Algorithm increases the first variable every time a further variable needs to go up, keeping degree constant
          # Successor of [0, 0, 0, a, b, c, d] is [a-1, 0, 0, 0, b+1, c, d]
            current_ordinal[sub_count+1] += 1
            temporary_variable = current_ordinal[sub_count]
            current_ordinal[sub_count] = 0 # This has to be the order to make sense if sub_count = 0
            current_ordinal[0] = temporary_variable - 1
            #print(' IF  Arriving here with count = '+str(count)+' and current ordinal is '+str(current_ordinal)) # For debugging
            # Should break out of the for loop, and go straight to the appending part
            break
        else:
          # Algorithm increases the degree by 1, and the power of the last variable goes to the first
          # Successor of [0, 0, 0, 0, a] is [a+1, 0, 0, 0, 0]
          # Case max_j = 1 falls here, with [a+1] succeeding [a]
          # Case max_j = 1 makes us alter a bit the code because max_j-1 = 0
          temporary_variable = current_ordinal[max_j-1]
          current_ordinal[max_j-1] = 0 # This has to be the order to make sense if count = 0
          current_ordinal[0] = temporary_variable+1
          #print('ELSE Arriving here with count = '+str(count)+' and current ordinal is '+str(current_ordinal))  # For debugging
    # We produce from 0 to max_ordinal, but only append (and output explanation) for the needed ones. This is the appending time.
    if count >= min_ordinal:
      dict_of_all_ordinals[count] = deepcopy(current_ordinal)
      list_of_all_ordinals.append(deepcopy(current_ordinal))
      #if explain:
        #print('{} produced, the {}{} of the ordinals in {} variables').format(current_ordinal, produced_so_far - 1, right_st_nd_rd_th(produced_so_far - 1), max_j)
  # Should be ready now, a list of ordinals (as lists) up to max_ordinal
  # We have now the return options
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'list':
    return list_of_all_ordinals
  elif formatted_output_as == 'dict':
    return dict_of_all_ordinals
  elif formatted_output_as == 'object':
    return dict_of_all_ordinals
  elif formatted_output_as == 'string':
    list_of_all_ordinals_as_strings = [str(ordinal) for ordinal in list_of_all_ordinals]
    return '\n'.join(list_of_all_ordinals_as_strings)
  else:
    return dict_of_all_ordinals

# Function generate_vectors_for_the_base
# Output_as being object, list should give the same thing
# Note that we write beta_j instead of u^d*beta_j to simplify things
# To not do so would be too much work and cause bugs to not do it
def generate_vectors_for_the_base(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'object'):
  r'''
  Returns the monomials which generate of (KU_p^hat)_{2d}(BU) whose degrevlex order for the monomials on beta_1, ..., beta_{max_j} are between min_ordinal and max_ordinal.
  '''
  # Note the following only depends on max_ordinal, max_j and min_ordinal
  # Nonetheless, we keep p, r and d as arguments for the moment
  # We never need more j's than as there are ordinals. We could insert the following:
  #if max_j > max_ordinal:
    #max_j = max_ordinal
  # Now we work on getting a canonical base for a good display. We recycle a line from create_table_bu() to create generators (each is a list)
  dict_of_ordinals_as_lists = produce_first_places(min_ordinal, max_ordinal, max_j, output_as = 'dict')
  # The following should be [min_ordinal..max_ordinal]
  list_of_keys = dict_of_ordinals_as_lists.keys()
  # For safety, let us establish a maximal degree for the monomials which appear
  # Since the degree is nondecreasing going from min_ordinal to max_ordinal
  bound_degree = sum(dict_of_ordinals_as_lists[max_ordinal])
  # Now we want to create nice strings from them. We will create a list of strings of them. (Also the GeneratorHomologyBU instances.)
  # There will be max_ordinal-min_ordinal+1 strings
  dict_of_generators_as_instances = {key:GeneratorHomologyBU(p, bound_degree, max_j, dict_of_ordinals_as_lists[key]) for key in list_of_keys}
  dict_of_generators_as_strings = {key:str(dict_of_generators_as_instances[key]) for key in list_of_keys}
  list_of_generators_as_strings = [dict_of_generators_as_strings[count] for count in range(min_ordinal, max_ordinal+1)]
  #if explain:
    #print('We compute {} ({} through {}) monomials on variables beta_1, ..., beta_{}'.format(max_ordinal-min_ordinal+1, right_st_nd_rd_th(min_ordinal), right_st_nd_rd_th(max_ordinal), max_j))
    #print('They are ordered as degrevlex (total degree then reverse lexicographic). They are:\n')
    #for count in range(min_ordinal, max_ordinal+1):
      #print('{} as the {} monomial (total degree {})').format(generators_as_strings[count-min_ordinal], right_st_nd_rd_th(count), sum(list_of_ordinals[count]))
    #print('')
  # We now format the output and add the output options
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'list':
    return list_of_generators_as_strings
  elif formatted_output_as == 'dict':
    return dict_of_generators_as_strings    
  elif formatted_output_as == 'object':
    return dict_of_generators_as_strings    
  elif formatted_output_as == 'string':
    return '\n'.join(list_of_generators_as_strings)
  else:
    return list_of_generators_as_strings

######################################################################
# CODE: MAIN ENGINE FOR COMPUTING THE MATRIX ASSOCIATED TO AN ADAMS OPERATION
######################################################################

# Function create_problem_matrix
# New version of function old_create_problem_matrix, currently deprecated
# This new version was made to consume way less computational power
# Helps with creating a square table (a table is also a matrix) of dimensions max_j by max_j (beta_0 ignored)
# Or max_j+1 by max_j+1 if we allow beta_0 by having min_j_0_1 == 0
# The solutions of this matrix are psi^r(beta_j), and are computed in the function create_table_cpinf()
# To account for dimension 2d of K-cohomology, we multiply all elements by r^{-d}
def create_problem_matrix(p, r, min_j_0_1, max_j, d, output_as = 'list'):
  # We first normalize to integer to accept True and False values as start_j_at_1
  # We start at 0 if we include beta_0, and at 1. No other choices.
  min_j_0_1 = ZZ(min_j_0_1)
  assert min_j_0_1 in [0, 1], 'min_j_0_1 has to be 0 or 1'
  # Should only work for max_j >= 1
  assert (max_j >= min_j_0_1), 'max_j needs to be at least min_j_0_1'
  matrix_size = max_j - min_j_0_1 + 1
  # from Sage import *
  # Do the following to mean that the variable t will work as a polynomial variable
  ring = PolynomialRing(ZZ, 1, 't')
  t = var('t')
  # The matrix is a list of lists, each of those lists represents a row and comes from the coefficients of a polynomial
  # The matrix should have max_j lists, each list of length max_j, corresponding to beta_1 through beta_{max_j}
  the_matrix_as_list = []
  # Here is the alteration over old_create_problem_matrix: we create a separate vector with the polynomials
  # So we don't need to compute the powers as many times, as they are stored when calculated the first time
  # We start with placeholder list (setting value is quicker than expanding)
  # The following lines will compute powers of (t+1)**r-1 from power 0 to power max_j
  list_of_computed_polynomials = [0]*(max_j+1)
  shortcut_polynomial = expand((t+1)**r-1)
  for count in range(0, max_j+1):
    if count == 0:
      list_of_computed_polynomials[count] = expand(1)
    elif count == 1:
      list_of_computed_polynomials[count] = shortcut_polynomial
    else:
      list_of_computed_polynomials[count] = expand(list_of_computed_polynomials[count-1]*(shortcut_polynomial))
  # Now we can use the polynomials without generating them every time
  # Note that the first polynomial, 1, will be discarded below if and only if min_j_0_1 == 1
  # But only later in this function. Initially, it belongs to the list the_matrix_as_list
  for count in range(min_j_0_1, max_j+1):
    poly = list_of_computed_polynomials[count]
    coefficients_list = poly.list()
    #if explain:
      #print('{}{} polynomial is {}'.format(count, right_st_nd_rd_th(count), poly))
      #print('with coefficients {}'.format(poly.list()))
    # We only want the coefficients from t^0 or t^1 through t^{max_j} (depending on min_j_0_1)
    row = coefficients_list[min_j_0_1:max_j+1]
    # If we don't have enough coefficients due to a low degree, we add 0's to row
    # The polynomial poly is of degree count*r, and we want to have up to coefficient of t^(max_j)
    if count*r < max_j:
      row += [0]*(max_j - count*r)
    #if explain:
      #print('Relevant coefficients are {}\n'.format(row))
    # Append it as a row to the matrix to store it
    the_matrix_as_list.append(row)
  # Now we multiply every entry of the matrix_size by matrix_size table by r^(-d) (effect on psi^r in cohomology dimension 2d)
  for row in range(0, matrix_size):
    for column in range(0, matrix_size):
      the_matrix_as_list[row][column] *= r**(-d)
  # Preparing the output
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'list':
    return the_matrix_as_list
  else:
    the_matrix_as_matrix = BasedMatrixZPHat(p, matrix_size, the_matrix_as_list)
    if formatted_output_as == 'object':
      return the_matrix_as_matrix
    if formatted_output_as == 'string':
      return str(the_matrix_as_matrix.matrix_display('M'))
    else:
      return the_matrix_as_matrix

# Function compute_adams_operation_of_single_beta
# For given max_j, produces psi^r(u^d*beta_{max_j}) as sum of other beta's of indices <= j
# Computes it only for max_j, in contrast to create_table_cpinf, which computes psi^r(beta_j) from 1 to max_j
def compute_adams_operation_of_single_beta(p, r, max_j, d, output_as = 'list'):
  r'''
  Returns psi^r of u^d*beta_{max_j} in K_{2d}(BU).
  '''
  # If max_j = 0 then we do something special, since the Adams operation is very easy to compute
  if max_j == 0:
    output_list = [0]
    output_vector = vector(output_list)
    output_object = output_vector
    output_string = '{}*{}{}0'.format(QQ(r**(d)), neat_u_power(d), SYMBOL_FOR_GENERATORS)
  else:
    # We get the max_j by max_j matrix (called A_{il} in the thesis) to compute the coefficients of psi^(beta_{max_j}) (called B_{max_jl} in the thesis)
    # Of course A_{il} is multiplied by r^{-d} which implies B_{ij} multiplied by r^{d}
    # Since we already solved the problem of beta_0 we can exclude it for the_matrix of create_problem_matrix
    the_matrix = create_problem_matrix(p, r, 1, max_j, d, output_as = 'object')
    the_vector_as_list = [0]*(max_j-1) + [1]
    the_vector = vector(the_vector_as_list)
    solution = the_matrix.matrix_display('M').solve_right(the_vector)
    # The solution will be a vector of length max_j, with B_{max_j,1} in position 0 and B_{max_j, max_j} in position max_j-1
    output_vector = solution
    output_object = output_vector
    output_list = list(output_vector)
    output_string = ' + '.join('{}*{}{}{}'.format(output_object[count - 1], neat_u_power(d), SYMBOL_FOR_GENERATORS, count) for count in range(1, max_j+1))
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'object':
    # This by default includes vector, as vectors as well are matrices are classified as objects for output_as purposes
    return output_object
  if formatted_output_as == 'list':
    return output_list
  if formatted_output_as == 'string':
    return output_string
  else:
    return output_object

# Function create_table_cpinf
# Specifies a specific Adams operation \psi_r, and a size max_j
# Describes how the operation acts on the first max_j+1 elements of K_0(CP^inf) = K_0(BU(1)), from beta_0 through beta_(max_j)
# Result is a square (max_j + 1)-matrix as \psi_r(\tilde(beta_j)) is a linear combination of \tilde(beta_i)'s with i <= j
# Specially, psi^r(beta_0) = beta_0, and the coefficient of beta_0 in every other psi^r(beta_j) is 0 (this is true for d=0 only)
# That means that the first row (corresponding to beta_0) is [1, 0, ..., 0], and the leftmost column has all 0's except for the top left corner which is 1.
# We need to solve max_j linear systems, all with the same matrix, obtained through create_matrix_cpinf
# It would be equivalent to invert the matrix (and then transpose). Likely it would not save time, so we keep it this way.
def create_table_cpinf(p, r, min_j_0_1, max_j, d, output_as = 'list'):
  r'''
  Returns the matrix of psi^r on a finite rank submodule of K_0(CP^infty).
  '''
  # min_j_0_1 must be 0 or 1, otherwise the submodule will not be closed under psi^r, and max_j has to be at least min_j_0_1
  min_j_0_1 = ZZ(min_j_0_1)
  assert min_j_0_1 in [0, 1], 'min_j_0_1 should be 0 or 1'
  assert (max_j >= min_j_0_1), 'max_j needs to be at least min_j_0_1'
  table_size = max_j - min_j_0_1 + 1
  # The following has dimensions max_j or max_j+1 depending on min_j_0_1
  # We start with create_problem_matrix. What we want are the coefficients of its inverse.
  the_matrix = create_problem_matrix(p, r, min_j_0_1, max_j, d, output_as = 'object')
  # This is already a Sage matrix
  # It is max_j by max_j or (max_j+1) by (max_j+1), depending on including beta_0 which is determined by min_j_0_1
  # The easiest way is to compute the transpose of the inverse
  # Transposed, each row j gives the coefficients of beta_l in psi^r(beta_j)
  # Writing the_matrix**(-1).transpose() will try to transpose -1! So we add parenthesis
  the_table_as_matrix = (the_matrix.matrix_display('M')**(-1)).transpose()
  the_table_as_list = list_from_matrix(the_table_as_matrix)
  #if explain:
    #for count in range(min_j_0_1, max_j+1):
      #print('Coefficients of psi^{}({}beta_{}) over the generators'.format(r, neat_u_power(d), count))
      #print('{}beta_{}, {}beta_{}, ..., {}beta_{}) are {}'.format(neat_u_power(d), min_j_0_1, neat_u_power(d), min_j_0_1+1, neat_u_power(d), the_table_as_list[count]))
    #print('Consequently, table for psi^{} on (KU_{}^hat)_{}(CP^inf) is:'.format(r, p, 2*d))
    #for row in the_table_as_list:
      #print(row)
    #print('')
  # The output options
  formatted_output_option = format_output_as(output_as)
  if formatted_output_option == 'object':
    return the_table_as_matrix
  elif formatted_output_option == 'list':
    # Recall option 'list' includes option 'table'
    return the_table_as_list
  elif formatted_output_option == 'string':
    # the_table_as_matrix is an element of MatrixSpace(QQ, table_size)
    return str(the_table_as_matrix)
  else:
    return the_table_as_matrix

# Function create_table_bu
# Specifies a specific Adams operation \psi^r, and a evaluation max_ordinal (later renamed to max_ordinal)
# Describes how the operation acts on the first elements of K_{2d}(BU(max_ev)) with evaluation <= max_ordinal
# We always remember that the ordinal position of a monomial is the same as the number of monomials before it
# It is also possible to specify a value max_j (to allow only from \beta_1 to \beta_{max_j})
# Result is a square matrix as \psi^r of an element is a linear combination of others with <= evaluation
def create_table_bu(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'object'):
  r'''
  Returns a table (a list of lists) where each line has the coefficients of the image under psi^r of a particular monomial generator of (KU_p^hat)_{2d}.
  '''
  # Not really used, but could be useful if we want to insert some asserts
  table_size = max_ordinal - min_ordinal + 1
  # Let's first obtain psi^r of beta_0, beta_1, beta_2, ... beta_(max_j). This will be our lookup table.
  # lookup_table and list_of_ordinals are both lists of lists
  list_of_ordinals = produce_first_places(min_ordinal, max_ordinal, max_j, output_as = 'list')
  # Even if beta_0 or u^d*beta_0 is not needed, we include it in the lookup table for uniformity
  lookup_table = create_table_cpinf(p, r, 0, max_j, d, output_as = 'list')
  # There is a method (for the class GeneratorHomologyBU) that computes psi^r for each monomial
  # First we create a list of length max_ordinal+1 in which each object is an instance of GeneratorHomologyBU
  # If we wanted to have minimal n_bounds for all, we would use sum(row) instead of sum(list_of_ordinals[-1])
  # But it is best to use sum(list_of_ordinals[-1]) as it is a bound after all, not the minimum (total degree) for each monomial
  max_degree_common_to_instances = sum(list_of_ordinals[-1])
  list_of_pre_images = [GeneratorHomologyBU(p, max_degree_common_to_instances, max_j, ordinal) for ordinal in list_of_ordinals]
  if len(list_of_pre_images) != table_size:
    print('Warning: For some reason the list of generators has the wrong size.')
  # Now we compute psi^r for each generator (using its method) and put it into a new list (which starts empty)
  # For each of them, it will be an ElementHomologyBU
  list_of_images = [pre_image.compute_adams_operation_of_monomial(lookup_table) for pre_image in list_of_pre_images]
  # Now we want to get the coefficients in terms of the possible monomials. We use a Sage native function.
  # Again we need the polynomial ring to extract the coefficients. We can use self.rat_ring for each instance
  a_rat_ring = PolynomialRing(QQ, max_j, string_of_good_variables(1, max_j+1), order = 'degrevlex')
  # [row.monomial_coefficient(rat_ring.monomial(0    ....        tupleordinal)) for ordinal in list_of_ordinals] should produce the coefficients in a list
  # starting from beta_0 until the ordinal in position max_ordinal
  # That goes for every polynomial from psi^r(beta_0) (if included) through psi^r(monomial in position max_ordinal)
  # This will always be a square table or size table_size (and includes beta_0 as first row if and only if includes beta_0 as first column)
  list_of_lists_of_coefficients = [[row.as_polynomial.monomial_coefficient(a_rat_ring.monomial(*ordinal)) for ordinal in list_of_ordinals] for row in list_of_images]
  # Now the outputs
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'list':
    return list_of_lists_of_coefficients
  elif formatted_output_as == 'object':
    return list_of_images
  else:
    if formatted_output_as == 'string':
      list_of_strings = ['{}psi^{}({}{}) = {}'.format(neat_u_power(-d), r, neat_u_power(d), list_of_pre_images[count], list_of_images[count]) for count in range(table_size)]
      return '\n'.join(list_of_strings)
    elif formatted_output_as == 'dict':
      dict_of_images = {count:list_of_images[count+1-min_ordinal] for count in range(min_ordinal, table_size+1)}
      return dict_of_images
    else:
      return list_of_images
  # Past comments on this function, if case they are ever useful:
  # We want to create a matrix with (max_ordinal)*(max_ordinal) entries, corresponding from 1 to max_ordinal (in this hypothesis we remove beta_0)
  # The entry of row i and column j should be the j-th coefficient accompanying the monomial in place i
  # when psi^r(monomial in place j) is written in terms of other monomials
  # Example: component of i = [2, 0, 1] (beta_1^2beta_3) from j = [3, 1, 1] (beta_1^3beta_2^1beta_3^1)
  # This is impossible, as degree is always preserved...
  # New example: component of i = [2, 0, 1] (beta_1^2beta_3) from j = [0, 0, 3] (beta_3^3)
  # Since psi^4(beta_3^3) is psi^4(beta_3)^3 = (7/128*beta_1 -3/64*beta_2 + 1/64*beta_3)^3
  # We can compute it by putting each of the 3^3 monomials which appear in its correct categories

# Function get_coefficient_from_matrix_of_cofactors()
# This function can compute the coefficients B_jl individually, using the cofactor formula
# It was originally designed to verify or disprove the conjecture that B_jl != for 1 <= l <= j
# It takes 2 minutes to compute all B_jl for j = 150; compute_adams_operation_of_single_beta does it in 2 seconds.
# It is efficient, though, to compute a single B_jl, or to study the cofactor matrix and its singular/nonsingularity.
def get_coefficient_from_matrix_of_cofactors(p, r, j, l, d, output_as = 'string'):
  r'''
  Returns B_(`j``l`), the coefficient accompanying beta_`l` of the image of beta_`j` under an Adams operation on K_(2d)CP^(inf).
  '''
  # We create the matrix A
  whole_table = create_problem_matrix(p, r, 0, j, d, output_as = 'list')
  whole_matrix = Matrix(QQ, j+1, j+1, whole_table)
  # B_jl comes from the cofactor C_jl, which is the determinant of the matrix formed by removing row j and column l
  # Since to the left of column l we can easily compute the determinant, we focus of the matrix of rows l through j-1 and columns l+1 through j
  restricted_table = [row[l+1: j+1] for row in whole_table[l:j]]
  restricted_matrix = Matrix(QQ, j-l, j-l, restricted_table)
  # We will temporarily use -1 to mean 0 (valuation would be +Infinity, which is no accepted as element of ZZ)
  table_of_evaluations = [[ZZ(entry).ord(r) if entry != 0 else -1 for entry in row] for row in restricted_table]
  matrix_of_evaluations = Matrix(ZZ, j-l, j-l, table_of_evaluations)
  #if explain:
    #print(restricted_matrix)
    #print('')
    #print(matrix_of_evaluations)
    # Note that the evaluations are not necessarily decreasing in each row; try, for example, (p, r, j, l, d) = (5, 2, 40, 21, 0)
  determinant_of_whole = whole_matrix.determinant()
  determinant_of_restricted = restricted_matrix.determinant()
  # Now we give it as the formula obtained after a little algebraic work. Call B_jl the right coefficient
  coefficient = (-1)**(j+l)*special_product([whole_table[count][count] for count in range(1, l)])*determinant_of_restricted/determinant_of_whole
  return coefficient

######################################################################
# CODE: FUNCTIONS GENERATE_RELEVANT_MATRIX IN PREPARATION FOR COKERNEL FINDING
######################################################################

# Function generate_relevant_z_p_hat_matrix
# Generates matrix for psi^r - 1
# We use other generate_relevant functions here
def generate_relevant_z_p_hat_matrix(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'object'):
  r'''
  Returns an instance of BasedMatrixZPHat corresponding to psi^r - 1 on (KU_p^hat)_2d(BU)
  restricted to generators in max_j variables and whose degrevlex rankings are between min_ordinal and max_ordinal.
  
  Generates matrix for homology in BU with diagonal already subtracted by identity to represent psi^r-1,
  and multiplied by a constant depending on d to avoid technical problems with finding cokernel or elementary divisors.
  We do the following. Let B^trans be the matrix from create_table_bu for d = 0
  For d >= 0, we compute the cokernel of B^trans - r**d.
  For d < 0, we compute the cokernel of r**(-d)*B^trans - 1
  '''
  # We want either the object (a BasedMatrixZPHat) or a string
  relevant_matrix = generate_relevant_z_p_hat_matrix_for_growing_d(p, r, min_ordinal, max_ordinal, max_j, [d], output_as = 'dict')[d]
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'object':
    return relevant_matrix
  elif formatted_output_as == 'string':
    return str(relevant_matrix)
  else:
    return relevant_matrix

# Function generate_relevant_z_p_hat_matrix_for_growing_strata
# Generates matrix for psi^r - 1 on the specified strata
def generate_revelant_z_p_hat_matrix_for_growing_strata(p, r, range_for_fixed_degree, max_j, d, output_as = 'object'):
  r'''
  Returns instances of BasedMatrixZPHat corresponding to psi^r - 1 on (KU_p^hat)_2d(BU) for some strata.
  
  SEE ALSO: generate_relevant_z_p_hat_matrix
  '''
  # Recall compute_degrevlex_rankings_for_strata returns a tuple (min_ordinal, max_ordinal)
  # generate_revelant_z_p_hat_matrix_for_growing_ordinals verifies the integrity of the data
  range_for_min_and_max_ordinals = [compute_degrevlex_rankings_for_strata(fixed_degree, max_j) for fixed_degree in range_for_fixed_degree]
  pre_dict_of_cokernels = generate_revelant_z_p_hat_matrix_for_growing_ordinals(p, r, range_for_min_and_max_ordinals, max_j, d, output_as = 'dict')
  # Now we format the dict to have the right indexes, for example, create a list or a string
  dict_of_cokernels = {fixed_degree:pre_dict_of_cokernels[(compute_min_ordinal(fixed_degree, max_j), compute_max_ordinal(fixed_degree, max_j))] for fixed_degree in range_for_fixed_degree}
  list_of_cokernels = [dict_of_cokernels[fixed_degree] for fixed_degree in range_for_fixed_degree]
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'dict':
    return dict_of_cokernels
  if formatted_output_as == 'list':
    return list_of_cokernels
  if formatted_output_as == 'string':
    return '\n\n'.join(list_of_cokernels)
  if formatted_output_as == 'object':
    return dict_of_cokernels
  else:
    return dict_of_cokernels  

# Function generate_relevant_z_p_hat_matrix_for_growing_ordinals
# Generates matrix for psi^r - 1 on increasing submodules of (KU_p^hat)_2d(BU)
def generate_revelant_z_p_hat_matrix_for_growing_ordinals(p, r, range_for_min_and_max_ordinals, max_j, d, output_as = 'object'):
  r'''
  Returns instances of BasedMatrixZPHat corresponding to psi^r - 1 on (KU_p^hat)_2d(BU) for a range of possible choices of min_ordinal and max_ordinal.
  
  SEE ALSO: generate_relevant_z_p_hat_matrix
  '''
  # We verify data integrity. Each pair is supposed to be (min_ordinal, max_ordinal)
  for pair in range_for_min_and_max_ordinals:
    assert isinstance(pair, tuple), 'Each pair should be a tuple.'
    assert len(pair) == 2, 'Each pair should have length 2.'
    assert check_if_number(pair[0], 'ZZ>=0'), 'The first item of the pair should be a nonnegative integer.'
    assert check_if_number(pair[1], 'ZZ>=0'), 'The second item of the pair should be a nonnegative integer.'
    assert pair[0] <= pair[1], 'The pair should be nondecreasing.'
  # We get a min and a max for the range
  min_min_ordinal = min(pair[0] for pair in range_for_min_and_max_ordinals)
  max_max_ordinal = max(pair[1] for pair in range_for_min_and_max_ordinals)
  # In respect to the psi^r - 1, we go through generate_relevant_z_p_hat_matrix
  # We start at min_min_ordinal so need to be very careful with indexing its rows and columns
  # Row and column indexed by 0 correspond to the generator or degrevlex ranking min_min_ordinal
  uncut_relevant_matrix = generate_relevant_z_p_hat_matrix(p, r, min_min_ordinal, max_max_ordinal, max_j, d)
  uncut_relevant_matrix_as_table = list_from_matrix(uncut_relevant_matrix.matrix_display('M'))
  # Now to cut up the important parts
  dict_of_relevant_matrices = {}
  list_of_relevant_matrices = []
  for pair in range_for_min_and_max_ordinals:
    # Recall table_size is always max_ordinal - min_ordinal + 1
    table_size = pair[1] - pair[0] + 1
    table_for_pair = [line[pair[0]-min_min_ordinal:pair[1]-min_min_ordinal+1] for line in uncut_relevant_matrix_as_table[pair[0]-min_min_ordinal:pair[1]-min_min_ordinal+1]]
    matrix_for_pair = BasedMatrixZPHat(p, table_size, table_for_pair, transpose = False)
    dict_of_relevant_matrices[pair] = matrix_for_pair
    list_of_relevant_matrices.append(matrix_for_pair)
  # Now the output
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'dict':
    return dict_of_relevant_matrices
  elif formatted_output_as == 'object':
    return dict_of_relevant_matrices
  elif formatted_output_as == 'list':
    return list_of_relevant_matrices
  elif formatted_output_as == 'string':
    return_string = '\n\n'.join(str(list_of_relevant_matrices))
  else:
    return dict_of_relevant_matrices

# Function generate_relevant_z_p_hat_matrix_for_growing_d
# Like generate_relevant_z_p_hat_matrix, but for multiple d
# Uses the fact that for varying d, all it is done is multiplying and dividing the matrices by r^d
#and this way saves computational time by inverting a matrix only once
# Despite the name, the dimension d doesn't really have to be growing. Any list suffices.
def generate_relevant_z_p_hat_matrix_for_growing_d(p, r, min_ordinal, max_ordinal, max_j, range_for_d, output_as = 'object'):
  r'''
  Returns matrices corresponding to psi^r - 1, multiple homological degrees at the same time.
  
  The matrices produced by this function for the negative homological dimensions are multiplied by r^(-d).
  
  SEE ALSO: generate_relevant_z_p_hat_matrix
  '''
  # First we normalize range_for_d to get a list
  range_for_d = get_range_from_number(range_for_d, start_at = None)
  # We more or less mimic the ideas in generate_relevant_z_p_hat_matrix()
  # We first get the matrix for d=0 from create_table_bu
  pre_table_bu = create_table_bu(p, r, 0, max_ordinal, max_j, 0, output_as = 'table')
  # We first cut the table to be only between min_ordinal and max_ordinal, both in columns as in rows
  table_size = max_ordinal - min_ordinal + 1
  table_bu = [line[min_ordinal:max_ordinal+1] for line in pre_table_bu[min_ordinal:max_ordinal+1]]
  # Now we get all of them. Remember that for dimension d, the relevant matrix is multiplied by r**d
  # We would do the multiplication, and then subtract 1 from the diagonal, but might rather do a different thing
  # This is too complex for a single dictionary comprehension... let's do slowly
  dict_of_relevant_matrices = {}
  list_of_relevant_matrices = []
  for d in range_for_d:
    # To avoid precision error, we will do the following.
    # If d < 0, we subtract r**(-d) times identity of the create_table_bu for d=0
    # If d >= 0, we multiply the create_table_bu obtained for d=0 by r**d and then subtract the identity
    # r is invertible and thus won't change the cokernel
    if d < 0:
      # We subtract r**(-d) off the diagonal (first creating a copy)
      table_bu_for_d = [[column for column in row] for row in table_bu]
      for count in range(table_size):
        table_bu_for_d[count][count] -= r**(-d)
    else:
      # We create a new table for each d, first multiplying by r**d
      table_bu_for_d = [[(r**d)*column for column in row] for row in table_bu]
      # We then subtract 1 from the diagonal
      for count in range(table_size):
        table_bu_for_d[count][count] -= 1
    # We create matrices, not tables, in this function. We also transpose it for the same reasons of generate_relevant_z_p_hat_matrix.
    matrix_bu_for_d = BasedMatrixZPHat(p, table_size, table_bu_for_d, transpose = True)
    # And append the list and update the dict
    dict_of_relevant_matrices[d] = matrix_bu_for_d
    list_of_relevant_matrices.append(matrix_bu_for_d)
  # Time of the output
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'dict':
    return dict_of_relevant_matrices
  elif formatted_output_as == 'object':
    return dict_of_relevant_matrices
  elif formatted_output_as == 'list':
    return list_of_relevant_matrices
  elif formatted_output_as == 'string':
    return_string = '\n\n'.join(str(list_of_relevant_matrices))
  else:
    return dict_of_relevant_matrices

# Function generate_approximations_to_relevant_matrix
# Those appoximations are Z/p, Z/p^2, ..., Z/p^(max_j), which approximate Z_p^hat
def generate_approximations_to_relevant_matrix(p, r, min_ordinal, max_ordinal, max_j, d, max_k, output_as = 'string'):
  r'''
  Returns a list of approximations, from Z_p^hat to Z/p^kZ, of the matrix describing psi - 1.
  
  SEE ALSO: generate_relevant_z_p_hat_matrix
  '''
  relevant_matrix = generate_relevant_z_p_hat_matrix(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'list')
  # To save computational time, we only do the diagonalization once (makes sense for d = 0...)
  relevant_matrix.make_M_diagonal()
  # Now we'll produce the approximations.
  list_of_approximations = relevant_matrix.range_of_modular_reductions(max_k)
  # We return a list. Position 0 has the completed matrix, and in positions 1 through max_k their reductions modulo p^k
  return list_of_approximations

# Function generate_approximations_to_cokernel_of_adams_operation
# It allows computations of the kernel when the matrix of psi - 1, typically called relevant_matrix around,
#have coefficients in the approximations Z/p^kZ (k ranging from 1 to max_k) which approximate Z_p^hat
# Reasoning: If the matrices of generate_approximations_to_relevant_matrix approximate the relevant_matrix,
#the cokernels of the matrices approximate the cokernel of relevant_matrix
def generate_approximations_to_cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, max_k, output_as = 'list', orders_only = False):
  r'''
  Returns the instances of BasedMatrixZPHat and BasedMatrixZPK corresponding to the reductions from Z_p^hat to Z_{p^k} of the cokernel of psi^r - 1 on (KU_p^hat)_{2d}(BU) restricted to max_ordinal and max_j.
  
  SEE ALSO: generate_relevant_z_p_hat_matrix()
  '''
  # We expect max_k to be a number. If given a list, we take the max and get the range (there is no point in using other intervals)
  if not check_if_number(max_k, ['ZZ']):
    max_k = max(max_k)
  # We produce the list [0, 1, ..., max_k], as a range
  range_for_k = range(max_k+1)
  # We generate a list with max_j+1 strings
  generators_as_strings = generate_vectors_for_the_base(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'list')
  # We generate a list with one instance of BasedMatrixZPHat and max_k instances of BasedMatrixZPK, totalizing max_k+1 objects
  list_of_instances = generate_approximations_to_relevant_matrix(p, r, min_ordinal, max_ordinal, max_j, d, max_k, output_as = 'list')
  # We generate a list with max_k+1 lists, each of those lists having a certain number of tuples
  list_of_cokernels_as_lists = [instance.produce_cokernel(canonical_base = generators_as_strings, output_as = 'list') for instance in list_of_instances]
  name_strings = ['']*(max_k+1)
  for k in range_for_k:
    name_strings[k] += 'Cokernel of (psi^{})-1 restricted to max_ordinal = {} and max_j = {}, '.format(r, max_ordinal, max_j)
    if k == 0:
      name_strings[k] += 'with coefficients in the whole {}:'.format(get_ring_of_coefficients_from_k(p, 0))
    else:
      name_strings[k] += 'approximated to coefficients in {}:'.format(get_ring_of_coefficients_from_k(p, k))
  # We also believe it is useful the following object as a dict instead of a list
  dict_of_cokernels_as_objects = {k: FinitelyGeneratedAbelianGroup(name_strings[k], list_of_cokernels_as_lists[k]) for k in range_for_k}
  # Now we prepare the output. Note that the object will be the dictionary of FinitelyGeneratedAbelianGroup instances
  formatted_output_as = format_output_as(output_as)
  # In the future we may abandon the following practice, or move to the start of the function as assertions
  if formatted_output_as not in ['dict', 'string', 'object', 'list']:
    print('Warning: output_as should be \'string\', \'object\', \'dict\' or \'list\'. Defaulted to \'object\'.')
    formatted_output_as = 'object'
  if formatted_output_as == 'dict':
    return dict_of_cokernels_as_objects
  elif formatted_output_as == 'object':
    return dict_of_cokernels_as_objects
  elif formatted_output_as == 'list':
    list_of_cokernels_as_objects = [dict_of_cokernels_as_objects[k] for k in range_for_k]
    return list_of_cokernels_as_objects
  elif formatted_output_as == 'string':
    # We generate a list with max_k+1 strings
    # Depending on the options, we will output different things. It could be a single string or a list of strings, defaulted to the second option.
    list_of_return_strings = ['']*(max_k+1)
    for k in range_for_k:
      # Optional stuff, when the user has formatted_output_as to be 'explain', and still only for k=0 which works in about Z_p^hat
      #if explain and k == 0:
        #list_of_return_strings[0] += 'Original Adams operation matrix (recovered from AMB):\n'
        #list_of_return_strings[0] += str(instance.AMB)
        #list_of_return_strings[0] += '\n\nMatrix A:\n'
        #list_of_return_strings[0] += str(instance.A)
        #list_of_return_strings[0] += '\n\nMatrix M:\n'
        #list_of_return_strings[0] += str(instance.M)
        #list_of_added_explanations[0] += '\n\nMatrix B:\n'
        #list_of_added_explanations[0] += str(instance.B)
        #list_of_added_explanations[0] += '\n\nNow the cokernels themselves:\n'
      # The above is optional. The following is the only thing that will always be added to list_of_return_strings[k]:
      if orders_only:
        printing_choice = 'name_and_orders'
      else:
        printing_choice = 'full'
      list_of_return_strings[k] += dict_of_cokernels_as_objects[k].print_flexibly(printing_choice)
    return '\n\n'.join(list_of_return_strings)

# Function find_pre_image_cpinf
# In this case, we don't need to overcomplicate. pre_image is with respect to psi**r - 1
# (Note that this is not pre_image with respect to psi**r, which would be obtainable straight from create_problem_matrix)
# For d=0, any beta_j will have a multiple which is a pre-image. For d != 0, not necessarily
# Only need to get the table up to the maximum of finding_pre_image_of we want
# Monomials are incredibly easier to generate, as are the coefficients (no multiplication of polynomials in a ring)
def find_pre_image_cpinf(p, r, finding_pre_image_of, d, output_as = 'string', move_to_qq = False):
  r'''
  Returns the pre-image by psi^r - 1 of a multiple of a generator of (KU_p^hat)_2d(BU).
  '''
  # If given a number as a range, make it into a list for uniformity
  # We start with iterable objects (such as tuples and lists)
  try:
    finding_pre_image_of = list(finding_pre_image_of)
  except TypeError:
    # In this case, we have a single object, and we make a list out of it
    finding_pre_image_of = [finding_pre_image_of]
  # If user entered "beta_j" instead of simply the numbers j in finding_pre_image_of (what would be preferred), we will eliminate the "beta_" part
  for entry in finding_pre_image_of:
    # If something was entered like beta_number or another kind of string, we want to keep only the number. So we will do:
    entry = ZZ(''.join([character for character in str(entry) if character.isdigit()]))
  # Let's register the maximum as a variable. It is the highest we need to go. It will perform as max_j and max_ordinal
  max_pre_image = max(finding_pre_image_of)
  # Now we get the matrix for psi^r - 1. Let's always include beta_0 (as min_ordinal or min_pre_image argument) as it might be useful
  # From the point of view of generate_relevant_z_p_hat_matrix, the max_j and max_ordinal arguments should coincide (that means working with CPinf)
  relevant_matrix = generate_relevant_z_p_hat_matrix(p, r, 0, max_pre_image, max_pre_image, d, output_as = 'object')
  # For a neater display we will let Sage do its formatting magic on multivariable polynomials, from beta_0 to beta_{max_pre_image}
  # (We will need it inside the loop.)
  rat_ring = PolynomialRing(QQ, max_pre_image+1, string_of_good_variables(0, max_pre_image+1))
  # Now we compute the pre-image of each beta_number by this matrix using a dictionary
  # For each key,we expect a tuple (multiple, image_of_multiple)
  dict_of_pre_images = {}
  # We also prepare nice strings for the vectors and for the return if formatted_output_as == 'string'
  dict_of_full_strings = {}
  for number in finding_pre_image_of:
    # We form the vector [0, 0...0 , 1, 0... 0]
    vector_in_formation = [0]*(max_pre_image+1)
    vector_in_formation[number] = 1
    vector_formed = vector(vector_in_formation)
    # We use interact_with_vector method of BasedMatrix
    # We want coefficients in Z_p_hat (even if displayed as rationals)
    # We use the original M, stored in self.AMB (in case we diagonalize it too soon; diagonalizing changes the bases)
    multiple, pre_image_of_multiple = relevant_matrix.interact_with_vector(vector_formed, direct_or_pre = 'pre', move_to_qq = False, use_original_M = True, transpose = False)
    dict_of_pre_images[number] = (multiple, pre_image_of_multiple)
    # We will prepare full strings for each (and record in a dict)
    # We will, however, say explicitly if beta_number is not in the image. This can be discriminated via dict_of_pre_images[number]
    if multiple != 0:
      # We write the vector dict_of_pre_images[d][1] as a nice string using Sage native conversions and formatting
      # Note that u**d, if present, will be only added later
      vector_as_string = sum([rat_ring('{}*{}{}'.format(dict_of_pre_images[number][1][count], SYMBOL_FOR_GENERATORS, count)) for count in range(0, max_pre_image+1)])
      dict_of_full_strings[number] = '{}(psi^{} - 1)^(-1)({}*{}{}{}) = {}'.format(neat_u_power(-d), r, multiple, neat_u_power(d), SYMBOL_FOR_GENERATORS, number, vector_as_string)
    else:
      # In this case, let's write that it doesn't belong to the image.
      # This happens when, for example, there is a Z_p_hat summand to the cokernel. Complicated to describe exactly
      dict_of_full_strings[number] = '{}{}{} is not (nor is any of its nonzero multiples) in the image of (psi^{} - 1)'.format(neat_u_power(d), SYMBOL_FOR_GENERATORS, number, r)
  # Ready for output
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'object':
    return dict_of_pre_images
  elif formatted_output_as == 'list':
    list_of_pre_images = [dict_of_pre_images[key] for key in dict_of_pre_images.keys()]
    return list_of_pre_images
  elif formatted_output_as == 'string':
    # Finally the return string
    return_string = '\n'.join([dict_of_full_strings[number] for number in finding_pre_image_of])
    return return_string
  else:
    return dict_of_pre_images

# Function cokernel_of_adams_operation
# If output_as == 'string', it outputs a single string having the cokernel (including a Z for beta_0 = 1)
# If output_as == 'object' or 'list', outputs a list of strings, one for each summand the cokernel (and ignores beta_0)
@parallel(NUMBER_OF_CORES)
def cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'object', orders_only = False):
  r'''
  Returns an approximation to the cokernel of psi^r-1 as a map of (KU_p^hat)_{2d}(BU) to itself.
  
  It is an approximation as we only look at the generators whose degrevlex ranking on max_j variables is between min_ordinal and max_ordinal.
  
  The cokernel from this is a sub-Z_p^hat-module of the cokernel of psi^r-1 on the whole (KU_p^hat)_{2d}(BU).
  
  For the results be correct, it is essential that these monomials form a submodule which is invariant with respect to psi^r-1.
  '''
  # First of all, we get out relevant matrix (the one we want to compute the cokernel of)
  # It is an instance of BasedMatrixZPHat
  relevant_matrix = generate_relevant_z_p_hat_matrix(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'object')
  # We also have an appropriate cokernel_name
  name_for_cokernel = 'Cokernel of (psi^{})-1 restricted to min_ordinal = {} and max_ordinal = {} and max_j = {} and coefficients in {}'.format(r, min_ordinal, max_ordinal, max_j, get_ring_of_coefficients_from_k(p, 0))
  # Now we prepare output_as.
  formatted_output_as = format_output_as(output_as)
  # Now we prepare the generators. Remember they are only used for orders_only == False
  if orders_only:
    generators_as_strings = None
  else:
    # Now we get a list of strings, generators_as_strings, to feed into core_cokernel_of_adams_operation
    generators_as_strings = generate_vectors_for_the_base(p, r, min_ordinal, max_ordinal, max_j, d, output_as = 'list')
  # We can sent the work to core_cokernel_of_adams_operation() to not have to write two functions
  returning_return = core_cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, generators_as_strings, relevant_matrix, output_as = 'object', orders_only = orders_only)
  # We trust that core_cokernel_of_adams_operation() will prepare the output nicely
  return returning_return

# Function core_cokernel_of_adams_operation
# Designed from the previously auxiliary function in cokernel_of_adams_operation_with_growing_ordinals and cokernel_of_adams_operation_with_growing_d
# To make those processes streamlined into a single, parallel function, following the DRY principle
# We assume we are given almost every information we need, and do the computationally heavy lifting here
# This works for starting at the start skipping beta_0 or not, or single stratum, or many, or whatever; it does not discriminate, as long as gets coherent inputs
@parallel(NUMBER_OF_CORES)
def core_cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, generators_as_strings, relevant_matrix, output_as = 'object', orders_only = False):
  r'''
  Finds a cokernel of an Adams operation using monomials with max_j variables and specific bounds on the degrevlex orders.
  
  SEE ALSO: cokernel_of_adams_operation()
  '''
  # We get the matrix size from min_ordinal and max_ordinal
  matrix_size = max_ordinal - min_ordinal + 1
  # Basic verification
  assert isinstance(relevant_matrix, BasedMatrixZPHat), 'relevant_matrix should be a BasedMatrixZPHat'
  assert relevant_matrix.n == matrix_size, 'relevant_matrix should have size matrix_size'
  # Now we create the relevant_matrix to compute the cokernel
  name_for_cokernel = 'Cokernel of (psi^{})-1 on (KU_{}^hat)_{}(BU) restricted to min_ordinal = {}, max_ordinal = {} and max_j = {}'.format(r, p, 2*d, min_ordinal, max_ordinal, max_j)
  if orders_only:
    # Note: if orders_only == False, we don't need given_generators_as_strings. Let's force the issue to prove a point.
    generators_as_strings = None
    cokernel_as_object = relevant_matrix.produce_cokernel_orders(output_as = 'object', cokernel_name = name_for_cokernel)
  else:
    # The following line includes a call to make_M_diagonal
    cokernel_as_object = relevant_matrix.produce_cokernel(canonical_base = generators_as_strings, output_as = 'object', cokernel_name = name_for_cokernel)
  # Explanations, currently not in use. Possibly for a future formatted_output_as == 'explain' option
  #if explain:
    #print('This matrix is invertible in Q, but not necessaily in Z_{}^hat.'.format(p))
    #print('To do it, we put it in a diagonal form, with increasing powers of {} in the diagonal.'.format(p))
    #print('We keep track of each row and column operation made to know which domain vector is sent to which codomain vector.')
    #print('For every {}^k in the diagonal, there is one Z/{}Z in the cokernel, generated by a codomain vector.'.format(p, p))
    #print('')
    #print('Diagonalized matrix of (psi^{})-1 over the monomials:'.format(r))
    #print('')
    #print(relevant_matrix.matrix_display('M'))
  # Now we prepare the output
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'object':
    return cokernel_as_object
  elif formatted_output_as =='dict':
    # In this case we produce a one-item dictionary, indexed by all arguments (p, r, min_ordinal, max_ordinal, max_j, d)
    return {(p, r, min_ordinal, max_ordinal, max_j, d): cokernel_as_object}
  elif formatted_output_as == 'string':
    # Recall __str__ is full printing of print_flexibly, including the name for the cokernel
    return str(cokernel_as_object)
  elif formatted_output_as == 'list':
    # Return list of orders if orders_only, and else list of (order, generator)
    # We can use attributes of cokernel_as_object which is a FinitelyGeneratedAbelianGroup instance 
    # (Note that often returning as a list means something else. But in this case it is a single object, so we vary.)
    if orders_only:
      return cokernel_as_object.list_of_orders
    else:
      return cokernel_as_object.list_of_summands
  else:
    return cokernel_as_object

# Similar to cokernel_of_adams_operation, but on strata specified by fixed_degree and d
# Despite having 'growing' in the name, we only require that we have a list, doesn't really need to be increasing
# Might be merged later with others in the future
def cokernel_of_adams_operation_with_growing_strata(p, r, range_for_fixed_degree, max_j, d, output_as = 'object', orders_only = False):
  r'''
  Returns the cokernels of $\psi^r - 1$ restricted to the stratum to the monomials on u^d*beta_1, ..., u^d*beta_j of degree fixed_degree.
  
  SEE ALSO: cokernel_of_adams_operation()
  '''
  # We prepare an adequate list
  range_for_fixed_degree = get_range_from_number(range_for_fixed_degree, start_at = None)
  # The most different part is to compute min_ordinal and max_ordinal for each of them
  dict_of_min_ordinal = {}
  dict_of_max_ordinal = {}
  for fixed_degree in range_for_fixed_degree:
    dict_of_min_ordinal[fixed_degree] = compute_min_ordinal(fixed_degree, max_j)
    dict_of_max_ordinal[fixed_degree] = compute_max_ordinal(fixed_degree, max_j)
  # If needed, we produce the generators_as_strings from min_ordinal to max_ordinal for each fixed_degree
  # We get the max of the max_ordinals, which is the max_ordinal corresponding to the highest fixed_degree
  # (It would be the same as taking the max among the values of dict_of_max_ordinal)
  max_max_ordinal = dict_of_max_ordinal[max(range_for_fixed_degree)]
  # We want to have all the generators starting from degrevlex ranking 0, even if later many will be cut out
  generators_from_zero = generate_vectors_for_the_base(p, r, 0, max_max_ordinal, max_j, d, output_as = 'list')
  # For each fixed_degree, the generators are different, so we have dictionary indexed on fixed_degree
  dict_of_generators = {}
  for fixed_degree in range_for_fixed_degree:
    if orders_only:
      dict_of_generators[fixed_degree] = None
    else:  
      dict_of_generators[fixed_degree] = generators_from_zero[dict_of_min_ordinal[fixed_degree]:dict_of_max_ordinal[fixed_degree]+1]
  # We get the matrix, and the submatrices through a single call.
  dict_of_matrices = generate_revelant_z_p_hat_matrix_for_growing_strata(p, r, range_for_fixed_degree, max_j, d, output_as = 'dict')
  # Finally we call core_cokernel_of_adams_operation, parallelized
  pre_dict = {'output_as': 'object', 'orders_only': orders_only}
  normalized_inputs = [((p, r, dict_of_min_ordinal[fixed_degree], dict_of_max_ordinal[fixed_degree], max_j, d, dict_of_generators[fixed_degree], dict_of_matrices[fixed_degree]), pre_dict) for fixed_degree in range_for_fixed_degree]
  generator = core_cokernel_of_adams_operation(normalized_inputs)
  # Now we need to do a kludge, first indexing by min_ordinal (found at [0][0][2]), and only later indexing by fixed_degree
  # Note that since all the others are fixed, min_ordinal increases with first_degree (same can be said aboud max_ordinal)
  pre_dict_of_cokernels = {gened[0][0][2]:gened[1] for gened in generator}
  dict_of_cokernels = {fixed_degree:pre_dict_of_cokernels[dict_of_min_ordinal[fixed_degree]] for fixed_degree in range_for_fixed_degree}
  list_of_cokernels = [dict_of_cokernels[fixed_degree] for fixed_degree in range_for_fixed_degree]
  # Time for outputting
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'object':
    return dict_of_cokernels  
  elif formatted_output_as == 'dict':
    return dict_of_cokernels
  elif formatted_output_as == 'list':
    return list_of_cokernels
  elif formatted_output_as == 'string':
    return '\n\n'.join([str(cokernel) for cokernel in list_of_cokernels])
  else:
    return dict_of_cokernels

# Function cokernel_of_adams_operation_with_growing_ordinals
# To compute Adams operations over a range of generators
# Idea is to see how diagonalized matrix and cokernel behave through them
# Allows for a range for both max_ordinal and max_j
# For longer explanations, use function cokernel_of_adams_operation
# Added min_ordinal, in the absence of a good notion of a possible range_for_min_ordinal
# Options of output_as: string, list, dict with keys (max_ordinal, max_j), object (the dict)
# For all of them, the underlying items are FinitelyGeneratedAbelianGroup instances
# (even string, which will call str() upon the instance)
def cokernel_of_adams_operation_with_growing_ordinals(p, r, min_ordinal, range_for_max_ordinal, range_for_max_j, d, output_as = 'string', orders_only = False):
  r'''
  Returns the cokernels of psi^r - 1 for multiple submodules in (KU_p^hat)_{2d}(BU).
  
  SEE ALSO: cokernel_of_adams_operation()
  '''
  # min_ordinal will be typically 0 or 1, but it doesn't need to be
  # We expect the ranges to be a list; we'll compute the cokernel for all elements in the combination
  # Let's do some formatting and type-cleaning first... We want to ensure we have lists.
  range_for_max_ordinal = get_range_from_number(range_for_max_ordinal, None)
  range_for_max_j = get_range_from_number(range_for_max_j, None)
  assert isinstance(range_for_max_ordinal, (list, tuple)), 'range_for_max_ordinal should be a list or tuple'
  assert isinstance(range_for_max_j, (list, tuple)), 'range_for_max_j should be a list or tuple'
  # Getting some bounds is useful
  max_ordinal_bound = max(1, *range_for_max_ordinal)
  max_j_bound = max(1, *range_for_max_j)
  range_of_min_and_max_ordinals = [(min_ordinal, max_ordinal) for max_ordinal in range_for_max_ordinal]
  # We start by defining dictionaries of generators depending on max_j (to later appropriately cut)
  pre_dict_of_generators = {}
  pre_dict_of_dicts_of_matrices = {}
  for max_j in range_for_max_j:
    # First the generators, to take cuts of
    pre_dict_of_generators[max_j] = generate_vectors_for_the_base(p, r, min_ordinal, max_ordinal_bound, max_j, d, output_as = 'list')
    # We can use a direct call of generate_revelant_z_p_hat_matrix_for_growing_ordinals to get the BasedMatrixZPHat instances
    pre_dict_of_dicts_of_matrices[max_j] = generate_revelant_z_p_hat_matrix_for_growing_ordinals(p, r, range_of_min_and_max_ordinals, max_j, d, output_as = 'dict')
  # Now we define dicts with keys depending on (max_ordinal, max_j)
  dict_of_generators = {}
  dict_of_matrices = {}
  for max_j in range_for_max_j:
    for max_ordinal in range_for_max_ordinal:
      # We get those by obtaining the first max_ordinal-min_ordinal+1 generators
      table_size = max_ordinal-min_ordinal+1
      # Recall the 0th generator of pre_dict_of_generators[max_j] has degrevlex ranking min_ordinal
      dict_of_generators[(max_ordinal, max_j)] = pre_dict_of_generators[max_j][0:table_size]
      # We can now call the matrices stored in pre_dict_of_dicts_of_matrices[max_j]
      # They will each be an instance of BasedMatrixZPHat
      dict_of_matrices[(max_ordinal, max_j)] = pre_dict_of_dicts_of_matrices[max_j][(min_ordinal, max_ordinal)]
  # We use core_cokernel_of_adams_operation to simplify stuff
  # We normalize the inputs to work nicely with parallel
  pre_dict = {'output_as': 'object', 'orders_only': orders_only}
  normalized_inputs = [((p, r, min_ordinal, max_ordinal, max_j, d, dict_of_generators[(max_ordinal, max_j)], dict_of_matrices[(max_ordinal, max_j)]), pre_dict) for max_ordinal in range_for_max_ordinal for max_j in range_for_max_j]
  # We create the generator to make use of parallel
  generator = core_cokernel_of_adams_operation(normalized_inputs)
  # We also create a dictionary to collect the parallel results, collecting the results from the generator
  # Note gened[0] is input, gened[1] output, and gened[0][0] is (p, r, min_ordinal, max_ordinal, max_j, d), and we want (max_ordinal, max_j)
  dict_of_cokernels = {(gened[0][0][3], gened[0][0][4]):gened[1] for gened in generator}
  for key in dict_of_cokernels:
    assert isinstance(dict_of_cokernels[key], FinitelyGeneratedAbelianGroup), 'dict_of_cokernels[key] should be a FinitelyGeneratedAbelianGroup'
  # Now we start serving all output options
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'object':
    return dict_of_cokernels
  elif formatted_output_as == 'dict':
    return dict_of_cokernels
  elif formatted_output_as == 'list':
    # First we sort the keys. We believe Sage's natural order will be lexicographic on (max_ordinal, max_j)
    # We want to first sort by max_j, then max_ordinal
    list_of_args_and_cokernels = []
    # Each item will be of the form (max_j, max_ordinal, cokernel)
    for max_j in range_for_max_j:
      for max_ordinal in range_for_max_ordinal:
        list_of_args_and_cokernels.append((max_j, max_ordinal, dict_of_cokernels[(max_ordinal, max_j)]))
    return list_of_args_and_cokernels
  elif formatted_output_as == 'string':
    returning_string = 'Fixed arguments: p = {}, r = {}\n'.format(p, r)
    for max_j in range_for_max_j:
      returning_string += '\nFor j between 1 and {}, and for min_ordinal = {}, each following max_ordinal produces the cokernel:\n\n'.format(max_j, min_ordinal)
      for max_ordinal in range_for_max_ordinal:
        if orders_only:
          piece_of_string = dict_of_cokernels[(max_ordinal, max_j)].print_flexibly('orders_only')
        else:
          piece_of_string = dict_of_cokernels[(max_ordinal, max_j)].print_flexibly('summands_only')
        returning_string += '{}: {}\n\n'.format(max_ordinal, piece_of_string)
    return returning_string

# Function cokernel_of_adams_operation_with_growing_d
# To compute Adams operations over multiple dimensions d
# In general min_ordinal should be 0 or 1, but not necessarily
def cokernel_of_adams_operation_with_growing_d(p, r, min_ordinal, max_ordinal, max_j, range_for_d, output_as = 'object', orders_only = False):
  r'''
  Returns the cokernels of psi^r - 1 on submodules in different degrees of (KU_p^hat)_{*}(BU).
  
  SEE ALSO: cokernel_of_adams_operation()
  '''
  # First we format range_for_d. We don't need an interval, nor we need it to start at a specific value
  range_for_d = get_range_from_number(range_for_d, start_at = None)
  # Note generators_as_strings only uses info from max_ordinal and max_j. We can set d = 0 (or any other number) without meaning anything
  generators_as_strings = generate_vectors_for_the_base(p, r, min_ordinal, max_ordinal, max_j, 0, output_as = 'list')
  # Now we prepare all the tables. All are a multiplication or division by r of each other, so we use generate_relevant_z_p_hat_matrix_for_growing_d
  # Right now, they don't have beta_0, and the identity has already been subtracted
  dict_of_relevant_matrices = generate_relevant_z_p_hat_matrix_for_growing_d(p, r, min_ordinal, max_ordinal, max_j, range_for_d, output_as = 'dict')
  # What should be done is the diagonalization, which is time-consuming. We will call core_cokernel_of_adams_operation
  # We normalize the inputs to work nicely with parallel
  pre_dict = {'output_as': 'object', 'orders_only': orders_only}
  normalized_inputs = [((p, r, min_ordinal, max_ordinal, max_j, d, generators_as_strings, dict_of_relevant_matrices[d]), pre_dict) for d in range_for_d]
  # We create the generator to make use of parallel
  generator = core_cokernel_of_adams_operation(normalized_inputs)
  # We also create a dictionary to collect the parallel results, collecting the results from the generator.
  # d is picked up by argument [0][0][5] in (p, r, min_ordinal, max_ordinal, max_j, d)
  dict_of_cokernels = {gened[0][0][5]:gened[1] for gened in generator}
  list_of_cokernels = [dict_of_cokernels[d] for d in range_for_d]
  # We prepare the output
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'list':
    return list_of_cokernels
  if formatted_output_as == 'string':
    if orders_only:
      output_string = '\n\n'.join([cokernel.print_flexibly('name_and_orders') for cokernel in list_of_cokernels])
    else:
      output_string = '\n\n'.join([cokernel.print_flexibly('full') for cokernel in list_of_cokernels])
    return output_string
  else:
    return dict_of_cokernels

######################################################################
# CODE: THE "STUDY" FUNCTIONS
######################################################################

# Function study_cokernels
# Studies the data in many ways
# Options for graphing, analyzing growth, in a systematic way
def study_cokernels(increasing_arg, static_args, range_of_keys, dict_of_cokernels, what_to_do, output_as):
  r'''
  Returns a formatted output for the 'study_increasing' family of functions.
  
  INPUTS:
  
  `increasing_arg`: One among `p`, `max_ordinal`, `max_j`, `d`, `fixed_degree` and `k`.
  
  `static_args`: Many arguments, which together with 'increasing_arg',
  determine uniquely the cokernels to be analyzed.
  
  `range_of_keys`: List of possible values for `increasing_arg` for our analysis.
  
  `dict_of_cokernels`: Cokernels for each value in `range_of_keys` as instances of FinitelyGeneratedAbelianGroup.
  
  `what_to_do`: Choices to which aspect of the cokernels should be analyzed, should be among `POSSIBLE_ACTIONS`.
  
  `output_as`: Choice for output. Can be requested as built-in types, like 'list', 'bool', 'dict', 'string',
  as 'object', which returns a pre-chosen object with minimal information loss,
  and as 'csv', 'txt', which outputs the results to computer files.
  
  OUTPUT:
  
  The result of the required analysis, output as requested.
  
  SEE ALSO:
  
  study_increasing_d()
  study_increasing_fixed_degree()
  study_increasing_k()
  study_increasing_max_j()
  study_increasing_max_ordinal()
  study_increasing_p()
  '''
  # Remember we can actually bring the whole function object with us as an argument, called origin_function
  # To get the arguments of that functions, use *args and **kwargs
  # Deciding if getting the generator or the list_of_list_of_logs as argument...
  # Logic is to get the generator, since, firstdy, it is th most primitive of the entities
  # Second reason is that taking the logs may be optional, third reason is that the generator carries the arguments too
  # Note: float is sage.rings.real_mpfr.RealLiteral for SageMath
  # Another note: for d == 0 we would expect to exclude beta_0, and we are deciding if we it up to the user or not
  # For other cases, it might be useful, so the user may include it in the right way
  # So log, sum and max functions should always think they are receiving positive integers
  assert isinstance(increasing_arg, str), 'increasing_arg should be a string'
  assert isinstance(static_args, tuple), 'static_args should be a tuple'
  assert isinstance(range_of_keys, list), 'range_of_keys should be a list'
  assert isinstance(dict_of_cokernels, dict), 'dict_of_cokernels should be a dict'
  assert isinstance(what_to_do, str), 'what_to_do should be a string'
  # We already call format_what_to_do because we will need it soon
  formatted_what_to_do = format_what_to_do(what_to_do)
  assert formatted_what_to_do in POSSIBLE_ACTIONS.values(), 'formatted_what_to_do should be in POSSIBLE_ACTIONS.values()'
  for key in range_of_keys:
    # In this function we will always assume we have a FinitelyGeneratedAbelianGroup
    assert isinstance(dict_of_cokernels[key], FinitelyGeneratedAbelianGroup), 'Each cokernel should be a FinitelyGeneratedAbelianGroup'
    # keys for 'p' are a tuple (p, r), for 'max_j' a tuple (min_ordinal, max_ordinal, max_j), and for others a number
    # For 'p', p determines r. For 'max_j', max_j determines min_ordinal and max_ordinal
    if increasing_arg == 'p':
      assert isinstance(key, tuple), 'Key of range_of_keys should be a tuple'
      assert len(key) == 2, 'Length of key for increasing_arg \'p\' should be 2'
      assert check_if_number(key[0], ['ZZ']), 'First entry (corresponding to p) of tuple should be an integer'
      assert check_if_number(key[1], ['ZZ']), 'Second entry (corresponding to r) of tuple should be an integer'
    elif increasing_arg == 'max_j':
      assert isinstance(key, tuple), 'Key of range_of_keys should be a tuple'
      assert len(key) == 3, 'Length of key for increasing_arg \'max_j\' should be 3'
      assert check_if_number(key[0], ['ZZ']), 'First entry (corresponding to min_ordinal) of tuple should be an integer'
      assert check_if_number(key[1], ['ZZ']), 'Second entry (corresponding to max_ordinal) of tuple should be an integer'
      assert check_if_number(key[2], ['ZZ']), 'Third entry (corresponding to max_j) of tuple should be an integer'
    elif increasing_arg == 'max_ordinal' or increasing_arg == 'k' or increasing_arg == 'd' or increasing_arg == 'fixed_degree':
      assert check_if_number(key, ['ZZ']), 'Key of range_of_keys should be an integer'
    else:
      raise ValueError('increasing_arg must be \'p\', \'max_ordinal\', \'max_j\', \'k\', \'d\' or \'fixed_degree\'.')
  assert len(dict_of_cokernels) == len(range_of_keys), 'Length of range_of_keys should match length of dict_of_cokernels'
  assert len(range_of_keys) == len(set(range_of_keys)), 'No repetitions allowed among the elements of range_of_keys'
  # min means the logic boolean operator "and" on a list... (to get "or" on list, use sum on the list instead)
  # all() and any() can be used for this purpose. And it is better because of lazy evaluation
  assert all([key in dict_of_cokernels for key in range_of_keys]), 'Each key in range_of_keys have should be a key in dict_of_cokernels'
  # Get range_of_keys in order (maybe not...not sure how Sage will compute order of pairs...)
  # Either way, whatever is the order here, it will appear in list_of_cokernels
  # That is, we sort the keys from the dictionary dict_of_cokernels from the list range_of_keys
  range_of_keys = sorted(range_of_keys) 
  # Now we build the "ordered (as explain)" list of cokernels
  list_of_cokernels = []
  for key in range_of_keys:
    list_of_cokernels.append(dict_of_cokernels[key])
  # Now we parse the arguments from static_args
  # Need function here to recover the arguments of a function
  # Can use function.__name__ and function.__code__ if we ever figure out how
  # We seize the opportunity to provide annoucement strings
  # Note that we already asserted increasing_arg will be a valid value
  if len(range_of_keys) > 0:
    # If increasing_arg == 'max_j' or == 'p', then range_of_keys is a list of tuples
    # To be able to better order them, we introduce the variable potentially_modified_range_of_keys
    # The three arguments of the tuple (min_ordinal_for_max_j[max_j], max_ordinal_for_max_j(max_j), max_j) grow together
    # Either way, max_j is the relevant, so it is only max_j that we list.
    if increasing_arg == 'max_j':
      potentially_modified_range_of_keys = [key[2] for key in range_of_keys]
    # For 'p', the tuple is (p, r), and the p is the relevant one as r is computed from p
    elif increasing_arg == 'p':
      potentially_modified_range_of_keys = [key[0] for key in range_of_keys]
    else:
      potentially_modified_range_of_keys = range_of_keys
    minimum_in_range_of_keys = min(potentially_modified_range_of_keys)
    maximum_in_range_of_keys = max(potentially_modified_range_of_keys)
    if potentially_modified_range_of_keys == list(range(minimum_in_range_of_keys, maximum_in_range_of_keys)):
      range_of_keys_as_shorter_string = 'interval [{}..{}]'.format(minimum_in_range_of_keys, maximum_in_range_of_keys)
    else:
      range_of_keys_as_shorter_string = 'subset of interval [{}..{}]'.format(minimum_in_range_of_keys, maximum_in_range_of_keys)
  else:
    range_of_keys_as_shorter_string = 'empty interval'
  if increasing_arg == 'p':
    min_ordinal, max_ordinal, max_j, d = static_args
    announcement_string = 'Fixed: min_ordinal = {}, max_ordinal = {}, max_j = {}, d = {}, while (p, r) are listed.'.format(*static_args)
    name_for_file_csv_txt = 'study_increasing_p({}, {}, {}, {}, {}, {})'.format(range_of_keys_as_shorter_string, min_ordinal, max_ordinal, max_j, d, formatted_what_to_do)
    top_row_first_column_csv = '"(p, r)"'
    end_top_row_second_column_csv = '(Fixed: min_ordinal = {}, max_ordinal = {}, max_j = {}, d = {})"'.format(*static_args)
  elif increasing_arg == 'max_ordinal':
    p, r, min_ordinal, max_j, d = static_args
    announcement_string = 'Fixed: p = {}, r = {}, min_ordinal = {}, max_j = {}, d = {}, while max_ordinal are listed.'.format(*static_args)
    name_for_file_csv_txt = 'study_increasing_max_ordinal({}, {}, {}, {}, {}, {}, {})'.format(range_of_keys_as_shorter_string, p, r, min_ordinal, max_j, d, formatted_what_to_do)
    top_row_first_column_csv = '"max_ordinal"'
    end_top_row_second_column_csv = '(Fixed: p = {}, r = {}, min_ordinal = {}, max_j = {}, d = {})"'.format(*static_args)
  elif increasing_arg == 'max_j':
    p, r, interval_of_degrees, d = static_args
    # We want, for the following strings, to write interval_of_degrees as SageMath interval [min..max]
    min_interval_of_degrees = min(interval_of_degrees)
    max_interval_of_degrees = max(interval_of_degrees)
    interval_of_degrees_as_string = 'interval [{}..{}]'.format(min_interval_of_degrees, max_interval_of_degrees)
    static_args_modified = (p, r, interval_of_degrees_as_string, d)
    announcement_string = 'Fixed: p = {}, r = {}, interval_of_degrees = {}, d = {}, while (min_ordinal, max_ordinal, max_j) are listed.'.format(*static_args_modified)
    name_for_file_csv_txt = 'study_increasing_max_j({}, {}, {}, {}, {}, {})'.format(range_of_keys_as_shorter_string, p, r, interval_of_degrees_as_string, d, formatted_what_to_do)
    top_row_first_column_csv = '"(min_ordinal, max_ordinal, max_j)"'
    end_top_row_second_column_csv = '(Fixed: p = {}, r = {}, interval_of_degrees = {}, d = {})"'.format(*static_args_modified)
  elif increasing_arg == 'k':
    p, r, min_ordinal, max_ordinal, max_j, d = static_args
    announcement_string = 'Fixed: p = {}, r = {}, min_ordinal = {}, max_ordinal = {}, max_j = {}, d = {}, while k are listed.'.format(*static_args)
    # Here range_of_keys could be '[0..{}]'.format(max(range_of_keys))
    # We use range_of_keys_as_shorter_string nonetheless (to keep uniformity)
    name_for_file_csv_txt = 'study_increasing_k([0..{}], {}, {}, {}, {}, {}, {}, {})'.format(range_of_keys_as_shorter_string, p, r, min_ordinal, max_ordinal, max_j, d, formatted_what_to_do)
    top_row_first_column_csv = '"k"'
    end_top_row_second_column_csv = '(Fixed: p = {}, r = {}, min_ordinal = {}, max_ordinal = {}, max_j = {}, d = {})"'.format(*static_args)
  elif increasing_arg == 'd':
    p, r, min_ordinal, max_ordinal, max_j = static_args
    announcement_string = 'Fixed: p = {}, r = {}, min_ordinal = {}, max_ordinal = {}, max_j = {}, while d are listed.'.format(*static_args)
    name_for_file_csv_txt = 'study_increasing_d({}, {}, {}, {}, {}, {}, {})'.format(range_of_keys_as_shorter_string, p, r, min_ordinal, max_ordinal, max_j, formatted_what_to_do)
    top_row_first_column_csv = '"d"'
    end_top_row_second_column_csv = '(Fixed: p = {}, r = {}, min_ordinal = {}, max_ordinal = {}, max_j = {})"'.format(*static_args)
  elif increasing_arg == 'fixed_degree':
    p, r, max_j, d = static_args
    announcement_string = 'Fixed: p = {}, r = {}, max_j = {}, d = {}, while fixed_degree are listed.'.format(*static_args)
    name_for_file_csv_txt = 'study_increasing_fixed_degree({}, {}, {}, {}, {}, {})'.format(range_of_keys_as_shorter_string, p, r, max_j, d, formatted_what_to_do)
    top_row_first_column_csv = '"d"'
    end_top_row_second_column_csv = '(Fixed: p = {}, r = {}, max_j = {}, d = {})"'.format(*static_args)
  # Now we produce all types of returning objects, depending on what is being asked
  # It is a waste of instructions, but a minimal one... and very needed one
  # We work one by one
  # Note that we always have the orders_only correctly set up in dict_of_cokernels so it is not of major importance here
  if formatted_what_to_do == 'nothing':
    returning_object = {}
    returning_dict = {}
    returning_list = []
    half_returning_string = ''
    start_top_row_second_column_csv = '"Nothing' # In this case there should be no file even
    second_to_last_row_csv = ''
  elif formatted_what_to_do == 'pure_cokernels':
    return_string = '\n'.join(['{}: {}'.format(key, cokernel) for key, cokernel in zip(range_of_keys, list_of_cokernels)])
    returning_object = dict_of_cokernels
    returning_dict = dict_of_cokernels
    returning_list = list_of_cokernels
    half_returning_string = return_string
    start_top_row_second_column_csv = '"Pure cokernels'
    second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, cokernel.print_flexibly('single_line')) for key, cokernel in zip(range_of_keys, list_of_cokernels)])
  elif 'logs' in formatted_what_to_do or formatted_what_to_do == 'assure_list_increases':
    # Note that we need to be careful if we are increasing p, as the prime will change for the logarithm each time
    # So let's implement both procedures (increasing_arg == p and the rest)
    if increasing_arg == 'p':
      # In this case we pass one list at a time to get_the_logs_maybe (since each will have its own prime p)
      # The prime p is in key[0] (each key is a tuple of the form (p, r))
      # We need to listify once with [] and delistify once with [0] (since get_the_logs_maybe uses other level of listing)
      dict_of_list_of_logs = {key:get_the_logs_maybe(p = key[0], bunch_of_cokernels = [dict_of_cokernels[key].list_of_summands])[0] for key in dict_of_cokernels}
      list_of_list_of_logs = [get_the_logs_maybe(p = key_pair[0], bunch_of_cokernels = [cokernel])[0] for key_pair, cokernel in zip(range_of_keys, list_of_cokernels)]
    else:
      dict_of_list_of_logs = get_the_logs_maybe(p, dict_of_cokernels, replace_orders_by_logs = True)
      list_of_list_of_logs = get_the_logs_maybe(p, list_of_cokernels, replace_orders_by_logs = True)
    if formatted_what_to_do == 'pure_logs':
      string_of_logs = '\n'.join(['{}: {}'.format(key, list_of_logs) for key, list_of_logs in zip(range_of_keys, list_of_list_of_logs)])
      returning_object = dict_of_list_of_logs
      returning_list = list_of_list_of_logs
      returning_dict = dict_of_list_of_logs
      half_returning_string = string_of_logs
      start_top_row_second_column_csv = '"Pure logs of orders of subgroups of cokernel'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, list_of_logs) for key, list_of_logs in zip(range_of_keys, list_of_list_of_logs)])
    elif formatted_what_to_do == 'max_of_logs':
      list_of_max_logs = [special_max(list_of_logs) for list_of_logs in list_of_list_of_logs]
      dict_of_max_logs = {key:special_max(dict_of_list_of_logs[key]) for key in dict_of_list_of_logs}
      string_of_max_logs = '\n'.join(['{}: {}'.format(key, max_log) for key, max_log in zip(range_of_keys, list_of_max_logs)])
      returning_object = dict_of_max_logs
      returning_list = list_of_max_logs
      returning_dict = dict_of_max_logs
      half_returning_string = string_of_max_logs
      start_top_row_second_column_csv = '"Max of logs of orders of subgroups of cokernel'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, max_log) for key, max_log in zip(range_of_keys, list_of_max_logs)])
    elif formatted_what_to_do == 'sum_of_logs':
      list_of_sum_logs = [special_sum(list_of_logs) for list_of_logs in list_of_list_of_logs]
      dict_of_sum_logs = {key:special_sum(dict_of_list_of_logs[key]) for key in dict_of_list_of_logs}
      string_of_sum_logs = '\n'.join(['{}: {}'.format(key, sum_log) for key, sum_log in zip(range_of_keys, list_of_sum_logs)])
      returning_object = dict_of_sum_logs
      returning_list = list_of_sum_logs
      returning_dict = dict_of_sum_logs
      half_returning_string = string_of_sum_logs
      start_top_row_second_column_csv = '"Sum of logs of orders of subgroups of cokernel'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, sum_log) for key, sum_log in zip(range_of_keys, list_of_sum_logs)])
    elif formatted_what_to_do == 'assure_list_increases':
      # Let's make instances of NonDecreasingIntegerList to illuminate the process of comparing two whole lists
      # Best to use logs here (nor orders nor cokernels), so that's what we do
      # Since we use this log, if formatted_output_as == 'string', we also display the list_of_logs
      string_of_logs = '\n'.join(['{}: {}'.format(key, list_of_logs) for key, list_of_logs in zip(range_of_keys, list_of_list_of_logs)])
      list_of_nondecreasingintegerlists = [NonDecreasingIntegerList(list_of_logs) for list_of_logs in list_of_list_of_logs]
      is_n_d_i_list_non_decreasing = check_monotonicity(list_of_nondecreasingintegerlists, '<=')
      returning_object = is_n_d_i_list_non_decreasing
      returning_bool = is_n_d_i_list_non_decreasing
      half_returning_string = 'List of logs of orders of subgroups increases? Answer: {}. See below:\n'.format(is_n_d_i_list_non_decreasing)
      half_returning_string += string_of_logs
      if is_n_d_i_list_non_decreasing:
        start_top_row_second_column_csv = '"Confirmation that the list of logs of orders of subgroups is indeed non decreasing'
      else:  
        start_top_row_second_column_csv = '"Confirmation that the list of logs of orders of subgroups decreases sometimes'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, list_of_logs) for key, list_of_logs in zip(range_of_keys, list_of_list_of_logs)])
  else:
    # In this case, p is irrelevant... so we don't care what we write
    # Let's write a "random" prime 43 to not have to worry with the case where increasing_arg == 'p' which would be confusing
    # Note that get_the_logs_maybe accepts both old way cokernels or FinitelyGeneratedAbelianGroup (it's the latter here) 
    dict_of_list_of_orders = get_the_logs_maybe(43, dict_of_cokernels, replace_orders_by_logs = False)
    list_of_list_of_orders = get_the_logs_maybe(43, list_of_cokernels, replace_orders_by_logs = False)
    if formatted_what_to_do == 'pure_orders':
      string_of_orders = '\n'.join(['{}: {}'.format(key, list_of_orders) for key, list_of_orders in zip(range_of_keys, list_of_list_of_orders)])
      returning_object = dict_of_list_of_orders
      returning_list = list_of_list_of_orders
      returning_dict = dict_of_list_of_orders
      half_returning_string = string_of_orders
      start_top_row_second_column_csv = '"Orders of subgroups of cokernel'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, list_of_orders) for key, list_of_orders in zip(range_of_keys, list_of_list_of_orders)])
    elif formatted_what_to_do == 'max_of_orders':
      list_of_max_orders = [special_max(list_of_orders) for list_of_orders in list_of_list_of_orders]
      dict_of_max_orders = {key:special_max(dict_of_list_of_orders[key]) for key in dict_of_list_of_orders}
      string_of_max_orders = '\n'.join(['{}: {}'.format(key, max_order) for key, max_order in zip(range_of_keys, list_of_max_orders)])
      returning_object = dict_of_max_orders
      returning_list = list_of_max_orders
      returning_dict = dict_of_max_orders
      half_returning_string = string_of_max_orders
      start_top_row_second_column_csv = '"Max of orders of subgroups of cokernel'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, max_order) for key, max_order in zip(range_of_keys, list_of_max_orders)])
    elif formatted_what_to_do == 'total_order':
      list_of_total_order = [special_product(list_of_orders) for list_of_orders in list_of_list_of_orders]
      dict_of_total_order = {key:special_product(dict_of_list_of_orders[key]) for key in dict_of_list_of_orders}
      string_of_total_order = '\n'.join(['{}: {}'.format(key, total_order) for key, total_order in zip(range_of_keys, list_of_total_order)])
      returning_object = dict_of_total_order
      returning_list = list_of_total_order
      returning_dict = dict_of_total_order
      half_returning_string = string_of_total_order
      start_top_row_second_column_csv = '"Total order of cokernel'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, total_order) for key, total_order in zip(range_of_keys, list_of_total_order)])
    elif formatted_what_to_do == 'assure_max_increases':
      list_of_max_orders = [special_max(list_of_orders) for list_of_orders in list_of_list_of_orders]
      string_of_max_orders = '\n'.join(['{}: {}'.format(key, max_order) for key, max_order in zip(range_of_keys, list_of_max_orders)])
      is_max_non_decreasing = check_monotonicity(list_of_max_orders, '<=')
      returning_object = is_max_non_decreasing
      returning_bool = is_max_non_decreasing
      half_returning_string = 'Max order among the orders of subgroups of cokernel increases? Answer: {}. See below:\n'.format(is_max_non_decreasing)
      half_returning_string += string_of_max_orders
      if is_max_non_decreasing:
        start_top_row_second_column_csv = '"Confirmation that the max order among the orders of subgroups of cokernel is indeed non decreasing'
      else:  
        start_top_row_second_column_csv = '"Confirmation that the max order among the orders of subgroups of cokernel decreases sometimes'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, max_order) for key, max_order in zip(range_of_keys, list_of_max_orders)])
    elif formatted_what_to_do == 'assure_sum_increases':
      # Note that corresponds to sum of logs, or multiplying the orders to get the total order
      list_of_total_order = [special_product(list_of_orders) for list_of_orders in list_of_list_of_orders]
      string_of_total_order = '\n'.join(['{}: {}'.format(key, total_order) for key, total_order in zip(range_of_keys, list_of_total_order)])
      is_total_non_decreasing =  check_monotonicity(list_of_total_order, '<=')
      returning_object = is_total_non_decreasing
      returning_bool = is_total_non_decreasing
      half_returning_string = 'Total order of cokernels increases? Answer: {}. See below:\n'.format(is_total_non_decreasing)
      half_returning_string += string_of_total_order
      if is_total_non_decreasing:
        start_top_row_second_column_csv = '"Confirmation that the max order among the orders of subgroups of cokernel is indeed non decreasing'
      else:  
        start_top_row_second_column_csv = '"Confirmation that the max order among the orders of subgroups of cokernel decreases sometimes'
      second_to_last_row_csv = '\n'.join(['"{}","{}"'.format(key, total_order) for key, total_order in zip(range_of_keys, list_of_total_order)])
  # Now we prepare the output of this function according to format_output_as(output_as)
  # Note that output_as == 'object' and 'string' should never fail (the others could)
  # To handle this, we opt for outputting a string if the requested output type is unavailable
  formatted_output_as = format_output_as(output_as)
  # Let's group the output string right now (whether is used or not)
  # It's good to leave this here because if there is any incompatibility, the returning_string is returned
  returning_string = announcement_string+'\n'+half_returning_string
  if formatted_output_as == 'object':
    try:
      return returning_object
    except NameError:
      print('\nWarning! what_to_do and output_as are incompatible. Returning as string instead.\n')
      return returning_string
  elif formatted_output_as == 'dict':
    try:
      return returning_dict
    except NameError:
      print('\nWarning! what_to_do and output_as are incompatible. Returning as string instead.\n')
      return returning_string
  elif formatted_output_as == 'list':
    try:
      return returning_list
    except NameError:
      print('\nWarning! what_to_do and output_as are incompatible. Returning as string instead.\n')
      return returning_string
  elif formatted_output_as == 'bool':
    try:
      return returning_bool
    except NameError:
      print('\nWarning! what_to_do and output_as are incompatible. Returning as string instead.\n')
      return returning_string
  # The two following are similar
  elif formatted_output_as == 'csv' or formatted_output_as == 'txt':
    # This is always defined... except that if we are ordered to do nothing, we create no file
    if formatted_what_to_do == 'nothing':
      return 'Nothing was done, as requested!'
    else:
      # We prepare the filename and contents. Since the name depends on time, we want to make sure they don't collide, so we force sleep
      # The limit for filenames is 255 bytes/chars. These will be typically around 100, 120 characters long, well within the limit.
      # We don't know how to control where the file appears. Right now, it appears at ~/ which is home directory
      time.sleep(2)
      pre_name_for_file = time.strftime(name_for_file_csv_txt+' at %Y.%m.%d %H.%M.%S', time.localtime())
      if formatted_output_as == 'csv':
        # Let's group the CSV strings for proper CSV output
        top_row_csv = top_row_first_column_csv+','+start_top_row_second_column_csv+' '+end_top_row_second_column_csv
        full_content_to_write = top_row_csv+'\n'+second_to_last_row_csv
        name_for_file = pre_name_for_file+'.csv'
      elif formatted_output_as == 'txt':
        full_content_to_write = returning_string
        name_for_file = pre_name_for_file+'.txt'
      # Now re do the same code for both
      with open(name_for_file, 'w+') as relevant_file:
        relevant_file.write(full_content_to_write)
      return 'Information successfully written to file {}'.format(name_for_file)
  elif formatted_output_as == 'string':
    # So if output_as is invalid, we return returning_string (this is different than the others...)
    return returning_string
  else:
    raise ValueError('Invalid choice for output_as')

# Function study_increasing_d
# Computes cokernel with fixed prime, primitive root, max_ordinal and max_j but increasing dimensions
# orders_only depends on what_to_do
def study_increasing_d(range_for_d, p, r, min_ordinal, max_ordinal, max_j, what_to_do, output_as = 'list'):
  r'''
  Studies the effect on the cokernel when we change the degree `d` of the homology.
  '''
  # To only use generators if requested:
  orders_only = orders_only_from_what_to_do(what_to_do)
  # We follow the model of the other "study" functions
  # First we take care of ensuring we have a range.
  range_for_d = get_range_from_number(range_for_d, start_at = None)
  assert len(range_for_d) >= 1, 'range_for_d should not be empty'
  # We can call cokernel_of_adams_operation_with_growing_d with this, which will do the parallel part
  dict_of_cokernels = cokernel_of_adams_operation_with_growing_d(p, r, min_ordinal, max_ordinal, max_j, range_for_d, output_as = 'dict', orders_only = orders_only)
  # We call study_cokernels, which will also be responsibly to formatting the return to our specification
  returning_return = study_cokernels('d', (p, r, min_ordinal, max_ordinal, max_j), range_for_d, dict_of_cokernels, what_to_do, output_as)  
  return returning_return

# Function study_increasing_fixed_degree()
# Keeps all other variables fixed, and change the degree of the monomials involved.
# In other words, changes the stratum (always bound by having at most max_j variables)
def study_increasing_fixed_degree(range_for_fixed_degree, p, r, max_j, d, what_to_do, output_as = 'object'):
  r'''
  Studies the effect on the cokernel when we change the stratum through a range of choices.
  '''
  # To only use generators if requested:
  orders_only = orders_only_from_what_to_do(what_to_do)
  # We format the range_for_fixed_degree as we want
  range_for_fixed_degree = get_range_from_number(range_for_fixed_degree, start_at = None)
  range_for_fixed_degree = [value for value in range_for_fixed_degree if value >= 0]
  assert len(range_for_fixed_degree) >= 1, 'range_for_fixed_degree should not be empty'
  # Not we obtain the dict_of_cokernels, as done in other study_increasing functions
  dict_of_cokernels = cokernel_of_adams_operation_with_growing_strata(p, r, range_for_fixed_degree, max_j, d, output_as = 'dict', orders_only = orders_only)
  returning_return = study_cokernels('fixed_degree', (p, r, max_j, d), range_for_fixed_degree, dict_of_cokernels, what_to_do, output_as)  
  return returning_return

# Function study_increasing_k
# Obtains only the decomposition of the cokernel, ignoring generators
# Works with growing max_k, that is, approximating from Z/(p^k)Z to Z_p^hat
# We don't allow ranges not in the form range(0, max_k+1) (it wouldn't make that much sense mathematically otherwise)
# And that's why we accept an integer instead of a list, and it computes from 0 (Z_p^hat) through max_k (Z/(p^k)Z)
# But list is also acceptable! We will make it into a range(0, max_k+1) though
def study_increasing_k(range_for_k, p, r, min_ordinal, max_ordinal, max_j, d, what_to_do, output_as = 'list'):
  r'''
  Studies the effect on the cokernel when we do approximations of Z_p^hat by Z_{P^k}.
  '''
  # To only use generators if requested:
  orders_only = orders_only_from_what_to_do(what_to_do)
  # First we want to guarantee that that whether a number or a list is fed as max_k, we get range(0, max(range_for_k)+1)
  # So if we don't start with a number (as it would be normal), we make it to be the maximum of that list/tuple/iterable
  # We can do it using get_range_from_number()
  range_for_k = get_range_from_number(range_for_k, start_at = None)
  max_k = max(range_for_k)
  # We need to ensure we have a list in range_for_k (study_cokernels() checks)
  range_for_k = list(range(0, max_k+1))
  assert len(range_for_k) >= 1, 'range_for_k should not be empty'
  # Note that for this one there is no gain in using parallel processing
  # Instead, we compute psi^r - 1 in Z_p^hat and then reduce them modulo Z/(p^k)Z, and only then the cokernels separately
  # Note that output_as == list obtains a list of the FinitelyGeneratedAbelianGroup instances, which is what we want
  # (The other study_increasing functions call for output_as == 'object'; this one is exception)
  list_of_cokernels = generate_approximations_to_cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, max_k, output_as = 'list', orders_only = orders_only)
  dict_of_cokernels = {each_k: each_cokernel for each_k, each_cokernel in zip(range_for_k, list_of_cokernels)}
  # Now we use study_cokernel
  returning_return = study_cokernels('k', (p, r, min_ordinal, max_ordinal, max_j, d), range_for_k, dict_of_cokernels, what_to_do, output_as)
  return returning_return

# Function study_increasing_max_j
# Obtains only the decomposition of the cokernel, ignoring generators
# Works with fixed degree, increasing number of betas/generators (and therefore of ordinals)
def study_increasing_max_j(range_for_max_j, p, r, interval_of_degrees, d, what_to_do, output_as = 'list'):
  r'''
  Studies the effect on the cokernel when we change the number of variables allowed.
  '''
  # To only use generators if requested:
  orders_only = orders_only_from_what_to_do(what_to_do)
  # We want to ensure we have a nonempty list of positive integers for range_for_max_j
  range_for_max_j = get_range_from_number(range_for_max_j, start_at = None)
  range_for_max_j = [value for value in range_for_max_j if value >= 1]
  assert len(range_for_max_j) >= 1, 'range_for_max_j should not be empty'
  # Given interval_of_degrees, find appropriate list of ordinals through compute_degrevlex_rankings_for_strata
  # (If not interval, simply takes max and min)
  # Pair each max_j with the correct max_ordinal. We will use dictionaries
  dict_of_tuples_of_min_and_max = {max_j:compute_degrevlex_rankings_for_strata(interval_of_degrees, max_j) for max_j in range_for_max_j}
  min_ordinal_for_max_j = {max_j:dict_of_tuples_of_min_and_max[max_j][0] for max_j in range_for_max_j}
  max_ordinal_for_max_j = {max_j:dict_of_tuples_of_min_and_max[max_j][1] for max_j in range_for_max_j}
  # Now we work towards invoking parallelized cokernel_of_adams_operation, starting with the input
  # We use output_as == 'object' as always
  pre_dict = {'output_as': 'object', 'orders_only': orders_only}
  # We need a list for study_cokernels. Remember that the order is "min_ordinal, max_ordinal, max_j"
  # (even if min_ordinal and max_ordinal are functions of max_j in this case)
  # These will be used for indexing when calling study_cokernels()
  min_and_max_ordinal_for_max_j_as_list_of_tuples = [(min_ordinal_for_max_j[key], max_ordinal_for_max_j[key], key) for key in max_ordinal_for_max_j]
  list_of_args = [(p, r, min_ordinal_for_max_j[max_j], max_ordinal_for_max_j[max_j], max_j, d) for max_j in range_for_max_j]
  normalized_input = [(args, pre_dict) for args in list_of_args]
  generator = cokernel_of_adams_operation(normalized_input)
  # We want as key (min_ordinal, max_ordinal, max_j) (3rd, 4th and 5th arguments) with value generated, this generated being ((input_args, input_kwargs), output)
  # We want the value to be simply the output
  dict_of_cokernels = {(generated[0][0][2], generated[0][0][3], generated[0][0][4]): generated[1] for generated in generator}
  # We plug study_cokernels here
  returning_return = study_cokernels('max_j', (p, r, interval_of_degrees, d), min_and_max_ordinal_for_max_j_as_list_of_tuples, dict_of_cokernels, what_to_do, output_as)
  return returning_return

# Function study_increasing_max_ordinal
# Obtains only the decomposition of the cokernel, ignoring generators
# Works with increasing (any set of them, really) ordinals, fixed max_j
def study_increasing_max_ordinal(range_for_max_ordinal, p, r, min_ordinal, max_j, d, what_to_do, output_as = 'list'):
  r'''
  Studies the effect on the cokernel when we change the number of monomials involved.
  '''
  # To only use generators if requested:
  orders_only = orders_only_from_what_to_do(what_to_do)
  # First, if we get a number instead of a list as range_for_max_ordinal, we would like to fix it.
  # We have no need of including ordinals lower than min_ordinal in range_for_max_ordinal
  range_for_max_ordinal = get_range_from_number(range_for_max_ordinal, start_at = None)
  # It throws an error if max_ordinal is less than min_ordinal. Because of that, we remove the zeros.
  range_for_max_ordinal = [value for value in range_for_max_ordinal if value >= min_ordinal]
  # We want nonempty
  assert len(range_for_max_ordinal) >= 1, 'range_for_max_ordinal should not be empty'
  # Now we start preparing the normalized input. Has to be of the form (*args, **kwars)
  pre_dict = {'output_as': 'dict', 'orders_only': orders_only}
  # We use cokernel_of_adams_operation_with_growing_ordinals to save time
  # The parallel processing will happen there, and we get a dict
  # We only need to reformat the dict it produces to have max_ordinals as keys instead of (max_ordinal, max_j)
  pre_dict_of_cokernels = cokernel_of_adams_operation_with_growing_ordinals(p, r, min_ordinal, range_for_max_ordinal, [max_j], d, **pre_dict)
  dict_of_cokernels = {max_ordinal:pre_dict_of_cokernels[(max_ordinal, max_j)] for max_ordinal in range_for_max_ordinal}
  # Plug this in study_cokernels
  returning_return = study_cokernels('max_ordinal', (p, r, min_ordinal, max_j, d), range_for_max_ordinal, dict_of_cokernels, what_to_do, output_as)
  return returning_return

# Function study_increasing_p
# Computes cokernel with fixed max_j and max_ordinal and increasing primes
# A primitive root of p^2, r, is computed for each and properly inserted
def study_increasing_p(range_for_p, min_ordinal, max_ordinal, max_j, d, what_to_do, output_as = 'list'):
  r'''
  Studies the effect on the cokernel when we change the prime p.
  '''
  # To only use generators if requested:
  orders_only = orders_only_from_what_to_do(what_to_do)
  # We normalize the argument to get a list (instead of a simgle number), and eliminate composite numbers
  # (The same operation will happen in get_primitive_roots_of_squares; we don't mind the repetition
  range_for_p = get_range_from_number(range_for_p, start_at = 3, force_odd_prime = True)
  # To avoid errors we forbid empty lists
  assert len(range_for_p) >= 1, 'range_for_p should not be empty'
  # We get a list of tuples (prime, primitive root of square) from 3 to max_p
  list_of_primes_and_roots = get_primitive_roots_of_squares(range_for_p, output_as = 'list')
  # Now we format those pairs to be (p, r) as arguments of cokernel_of_adams_operation
  # Making sure that they are formatted to be done in parallel
  pre_dict = {'output_as': 'object', 'orders_only': orders_only}
  # Here we tried to use *pair_of_args but some mysterious reason it doesn't work... so we do the cavemen way, with indexing
  normalized_input = [((pair_of_args[0], pair_of_args[1], min_ordinal, max_ordinal, max_j, d), pre_dict) for pair_of_args in list_of_primes_and_roots]
  # We create the generator (which has parallel computations)
  generator = cokernel_of_adams_operation(normalized_input)
  # We collect the results of this generator in a dict
  # Remember p and r are the 1st and 2nd positions
  dict_of_cokernels = {(gened[0][0][0], gened[0][0][1]): gened[1] for gened in generator}
  # We invoke study_cokernels to do what we want and then nicely display the output
  returning_return = study_cokernels('p', (min_ordinal, max_ordinal, max_j, d), list_of_primes_and_roots, dict_of_cokernels, what_to_do, output_as)
  return returning_return

######################################################################
# CODE: DEMOS, SCRIPTS AND EXECUTABLE CODE
######################################################################

# Function script_proving_coefficients_nonzero()
# Script to test if the coefficients of the matrix table_cpinf, B_jl, are nonzero for j >= 1 and 1 <= l <= j
# We know that B_jl is 0 for l > j and nonzero for l = j, but we will count those with l = j too.
def script_proving_coefficients_nonzero(data_type, data_on_primes, max_j, d, output_as = 'string'):
  r'''
  Returns whether the entries B_{jl} are nonzero 0 for 1 <= l <= j <= `max_j`
  
  (Verified nonzero for primes up to 60 and `max_j` <= 400.)
  '''
  # If data_type == 'max_prime', we will compute primes p up to that point, and find the corresponding r
  # If data_type == 'interval_primes', we will compute p and corresponding r's for primes inside interval
  # If data_type == 'p_and_r', we take those 2 only as p and r
  if data_type == 'max_prime':
    list_of_p_and_r = get_primitive_roots_of_squares(data_on_primes, output_as = 'list')
  elif data_type == 'interval':
    pre_list_of_p_and_r = get_primitive_roots_of_squares(data_on_primes[1], output_as = 'list')
    list_of_p_and_r = [line for line in pre_list_of_p_and_r if line[0] >= data_on_primes[0]]
  elif data_type == 'p_and_r':
    # We could use args and kwargs here, but probably not worth the hassle.
    list_of_p_and_r = list(data_on_primes)
  else:
    list_of_p_and_r = []
  # Function, or sub-function, aux_function_detect_nonzero
  # Due to the processing power needed we will do this in parallel
  @parallel(NUMBER_OF_CORES)
  def aux_function_detect_nonzero(p, r, max_j, d, output_as = 'string'):
    aux_output_boolean_all_nonzero = True
    aux_output_string = ''
    aux_output_string += 'For prime p = {} and topological generator r = {}:\n'.format(p, r)
    the_matrix = create_table_cpinf(p, r, 1, max_j, d, output_as = 'object')
    counting_nonzero = 0
    # We have a matrix of size max_j by max_j with the coefficients B_jl with 1 <= j, l <= max_j.
    # We expect the numbers below the diagonal to be nonzero. We check there are the right number of nonzero numbers there.
    expected_counting = (max_j)*(max_j+1)/2
    for row in range(0, max_j):
      for col in range(0, row+1):
        if the_matrix[row][col] == 0:
          aux_output_string += 'We found a 0 for row = {} and col = {}.\n'.format(row, col)
          aux_output_boolean_all_nonzero = False
        else:
          counting_nonzero += 1
    if counting_nonzero == expected_counting:
      aux_output_string += 'We counted {} nonzero entries, exactly as expected.'.format(counting_nonzero)
    else:
      aux_output_string += 'Of the {} tested entries, {} were zero and {} were nonzero.'.format(expected_counting, expected_counting-counting_nonzero, counting_nonzero)
    # Note that this output_as should be the same as the parent function.
    formatted_output_as = format_output_as(output_as)
    if formatted_output_as == 'bool':
      return aux_output_boolean_all_nonzero
    elif formatted_output_as == 'string':
      return aux_output_string
    elif formatted_output_as == 'object':
      return aux_output_boolean_all_nonzero
    else:
      return aux_output_boolean_all_nonzero
  # We normalize the inputs to work nicely with parallel
  pre_dict = {'output_as': output_as}
  normalized_inputs = [((pair[0], pair[1], max_j, d), pre_dict) for pair in list_of_p_and_r]
  # We create the generator to make use of parallel
  generator = aux_function_detect_nonzero(normalized_inputs)
  # We also create a dictionary to collect the parallel results, collecting the results from the generator
  # It can be either of strings or booleans, but the syntax is the same
  formatted_output_as = format_output_as(output_as)
  if formatted_output_as == 'string':
    dict_of_strings = {gened[0][0][0]:gened[1] for gened in generator}
    full_return_string = ''
    for pair in list_of_p_and_r:
      full_return_string += str(dict_of_strings[pair[0]])+'\n\n'
    return full_return_string
  else:
    dict_of_booleans = {gened[0][0][0]:gened[1] for gened in generator}
    # all() returns True only and only if every boolean inside is True
    full_boolean = all([dict_of_booleans[pair[0]] for pair in list_of_p_and_r])

# Function script_proving_cokernels_increase()
# Verifies if cokernels increase. Read once, and yield true for each prime it tried
# In principle outputs only to csv files, to keep the results safe.
# Didn't try all because it takes around 30 minutes for each prime (if going up to 500)
def script_proving_cokernels_increase(max_prime, min_ordinal_0_1, max_ordinal, max_j, d):
  r'''
  Returns CSV files which tell whether the cokernels of psi^r - 1 increase as we increase the domain and codomain.
  
  (It can be proved mathematically that the sequence of cokernels is always non-decreasing.)
  '''
  assert min_ordinal_0_1 in [0, 1], 'min_ordinal_0_1 should be 0 or 1'
  starting_time = time.time()
  print('This is the al_demo, which tries to assure if the cokernels always increase as lists.')
  print('\nComputing. It may take some time. Don\'t close this window...\n')
  range_for_max_ordinal = range(1, max_ordinal+1)
  for p in range(3, max_prime+1):
    if is_prime(p):
      correct_r = find_primitive_root_of_power(p, 2)
      print('Working on prime {}, which has primitive root {}'.format(p, correct_r))
      print(study_increasing_max_ordinal(range_for_max_ordinal, p, correct_r, min_ordinal_0_1, max_j, d, what_to_do = 'al', output_as = 'csv'))
      finish_time = time.time()
      print('Time elapsed: {} seconds.'.format(finish_time-starting_time))

# Function demo
# To demonstrate what the program Kobumuj can do
def demo():
  r'''
  Demonstrates what Kobumuj can do by executing many of its main functions.
  '''
  # We set a timer
  starting_time = time.time()
  print('----------------------D-E-M-O-N-S-T-R-A-T-I-O-N-----------------------')
  number_of_linebreaks = 3
  number_of_dashes = 70
  # First we call a few functions about obtaining the cokernels
  # We choose numbers which showcase the interesting properties of the subject
  # (Without taking too long)
  d = 3
  p = 5
  r = 2
  min_ordinal = 0
  max_ordinal = 70  
  range_for_max_ordinal = [1..40]
  range_for_max_j = [3..5]
  max_k = 10
  max_j = 9
  output_as = 'string'
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function cokernel_of_adams_operation:\n')
  print(cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function cokernel_of_adams_operation_with_growing_ordinals:\n')
  print(cokernel_of_adams_operation_with_growing_ordinals(p, r, min_ordinal, range_for_max_ordinal, range_for_max_j, d, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function generate_approximations_to_cokernel_of_adams_operation:\n')
  print(generate_approximations_to_cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, max_k, output_as))
  # Now a few other miscellaneous functions
  finding_pre_image_of = 95
  move_to_qq = False
  range_for_p = [3..50]
  data_type = 'max_prime'
  data_on_primes = 100
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function compute_adams_operation_of_single_beta:\n')
  print(compute_adams_operation_of_single_beta(p, r, max_j, d, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function find_pre_image_cpinf:\n')
  print(find_pre_image_cpinf(p, r, finding_pre_image_of, d, output_as, move_to_qq))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function get_primitive_roots_of_squares:\n')
  print(get_primitive_roots_of_squares(range_for_p, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function script_proving_coefficients_nonzero:\n')
  print(script_proving_coefficients_nonzero(data_type, data_on_primes, max_j, d, output_as))
  # Some arguments for the study_increasing functions, when needed
  what_to_do = 'pure_logs'
  min_ordinal_study = 1
  range_for_d = [-1, 2, -4, 0]
  range_for_fixed_degree = [1..2]
  range_for_k = [0..15]
  range_for_max_j_study = [1..15]
  range_for_max_ordinal_study = [1..60]
  range_for_p = [3..25]
  interval_of_degrees = [2..2]
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function study_increasing_d:\n')
  print(study_increasing_d(range_for_d, p, r, min_ordinal_study, max_ordinal, max_j, what_to_do, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function study_increasing_fixed_degree:\n')
  print(study_increasing_fixed_degree(range_for_fixed_degree, p, r, max_j, d, what_to_do, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function study_increasing_k:\n')
  print(study_increasing_k(range_for_k, p, r, min_ordinal_study, max_ordinal, max_j, d, what_to_do, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function study_increasing_max_j:\n')
  print(study_increasing_max_j(range_for_max_j, p, r, interval_of_degrees, d, what_to_do, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function study_increasing_max_ordinal:\n')
  print(study_increasing_max_ordinal(range_for_max_ordinal_study, p, r, min_ordinal_study, max_j, d, what_to_do, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function study_increasing_p:\n')
  print(study_increasing_p(range_for_p, min_ordinal_study, max_ordinal, max_j, d, what_to_do, output_as))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  finish_time = time.time()
  print('Time elapsed for demo: {} seconds.\n'.format(finish_time-starting_time))

# Function old_demo()
# First version of a demo which demonstrates what the program Kobumuj can do
def old_demo():
  r'''
  Demonstrates what kobumuj can do by executing many of its main functions. Superseded by function `demo()`.
  '''
  # We set a timer
  starting_time = time.time()
  print('-----------------O-L-D---D-E-M-O-N-S-T-R-A-T-I-O-N------------------')
  number_of_linebreaks = 3
  number_of_dashes = 70
  # Now we set up some argument values and perform calls of functions
  d = 3
  p = 5
  r = 2
  min_ordinal = 0
  max_ordinal = 60
  range_for_max_ordinal = range(1, 45)
  max_j = 9
  range_for_max_j = [3,4]
  max_k = 30
  output_as = 'string'
  orders_only = False
  move_to_qq = False
  finding_pre_image_of = 85
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function cokernel_of_adams_operation:\n')
  print(cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, output_as, orders_only))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function generate_approximations_to_cokernel_of_adams_operation:\n')
  print(generate_approximations_to_cokernel_of_adams_operation(p, r, min_ordinal, max_ordinal, max_j, d, max_k, output_as, orders_only))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function cokernel_of_adams_operation_with_growing_ordinals:\n')
  print(cokernel_of_adams_operation_with_growing_ordinals(p, r, min_ordinal, range_for_max_ordinal, range_for_max_j, d, output_as, orders_only))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  print('Function find_pre_image_cpinf:\n')
  print(find_pre_image_cpinf(p, r, finding_pre_image_of, d, output_as, move_to_qq))
  print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
  finish_time = time.time()
  print('Time elapsed for old_demo: {} seconds.'.format(finish_time-starting_time))

# Function debug_demo()
# A suite of tests with functions with different arguments
# We are satisfied every time this doesn't raise an error
# Keep adding tests as script gets more functions and features
def debug_demo():
  r'''
  Demo for debugging. Tests many functions to see if any errors arise.
  '''
  starting_time = time.time()
  d = 3
  p = 5
  max_p = 15
  r = 2
  max_j = 4
  max_ordinal = 20
  range_for_d = [d, d+1]
  range_for_fixed_degree = [0, 1, 3]
  range_for_max_j = [max_j, max_j+1]
  small_range_for_max_ordinal = [max_ordinal.. max_ordinal+1]
  large_range_for_max_ordinal = [max_ordinal.. max_ordinal+max_j]
  range_for_k = [0..10]
  range_for_p = [3..20]
  interval_of_degrees = [2..3]
  print('This is the debugging demonstration. If it runs to the end, we\'re happy.\n')
  for mutable_output_as in ['list', 'object', 'dict', 'string', 'bool']: # csv and txt options create too many files
    for mutable_min_ordinal in [0, 1]:
      for mutable_orders_only in [True, False]:
        print(cokernel_of_adams_operation(p, r, mutable_min_ordinal, max_ordinal, max_j, d, mutable_output_as))
        print(cokernel_of_adams_operation_with_growing_ordinals(p, r, mutable_min_ordinal, max_ordinal, max_j, d, mutable_output_as))
        print(cokernel_of_adams_operation_with_growing_ordinals(p, r, mutable_min_ordinal, small_range_for_max_ordinal, range_for_max_j, d, mutable_output_as))
        print(cokernel_of_adams_operation_with_growing_d(p, r, mutable_min_ordinal, max_ordinal, max_j, [d,d+1], mutable_output_as))
        print(cokernel_of_adams_operation_with_growing_strata(p, r, range_for_fixed_degree, max_j, d, mutable_output_as))
    for mutable_what_to_do in ['nothing', 'pc', 'po', 'mo', 'to', 'pl', 'ml', 'sl', 'al', 'am', 'as']:
      print(study_increasing_fixed_degree(range_for_fixed_degree, p, r, max_j, d, mutable_what_to_do, mutable_output_as))
      print(study_increasing_max_j(range_for_max_j, p, r, interval_of_degrees, d, mutable_what_to_do, mutable_output_as))
      for mutable_min_ordinal in [0, 1]:
        print(study_increasing_d(range_for_d, p, r, mutable_min_ordinal, max_ordinal, max_j, mutable_what_to_do, mutable_output_as))
        print(study_increasing_k(range_for_k, p, r, mutable_min_ordinal, max_ordinal, max_j, d, mutable_what_to_do, mutable_output_as))
        print(study_increasing_max_ordinal(large_range_for_max_ordinal, p, r, mutable_min_ordinal, max_j, d, mutable_what_to_do, mutable_output_as))
        print(study_increasing_p(range_for_p, mutable_min_ordinal, max_ordinal, max_j, d, mutable_what_to_do, mutable_output_as))
  list_of_lists = [[3,5,6], [3,4,7], [1,1,1,1,1,1,1], [0,1,3], [], [3,4,4], [7], [8], [2, 8], [+oo], [3, +oo], [1, +oo, +oo]]
  list_of_instances = [NonDecreasingIntegerList(each_list) for each_list in list_of_lists]
  for instance_1 in list_of_instances:
    for instance_2 in list_of_instances:
      if instance_1 < instance_2:
        print('{} < {}'.format(instance_1, instance_2))
      if instance_1 <= instance_2:
        print('{} <= {}'.format(instance_1, instance_2))
      if instance_1 > instance_2:
        print('{} > {}'.format(instance_1, instance_2))
      if instance_1 >= instance_2:
        print('{} >= {}'.format(instance_1, instance_2))
  print('\nSuccess!!!!\n')
  finish_time = time.time()
  print('Time elapsed for debug_demo: {} seconds.'.format(finish_time-starting_time))

# The following is executable, procedural code
# We use the execution (whether by being run directly or loaded by a module) to set up variables
# This is before the trigger for autorun of the demos, as it has to be
AUTORUN_MAXIMIZE_NUMBER_OF_CORES = False
if AUTORUN_MAXIMIZE_NUMBER_OF_CORES:
  NUMBER_OF_CORES = find_cpu_count()
SAFE_BASESTRING = safe_basestring()

# The following is executable, procedural code
# This is the code which will execute when the script is run directly. If imported from another file as a module, it doesn't.
# The if __name__ == '__main__' condition does the discrimination
AUTORUN_OLD_DEMO = False
AUTORUN_DEMO = False
AUTORUN_DEBUG_DEMO = False
if __name__ == '__main__':
  if AUTORUN_OLD_DEMO:
    old_demo()
  if AUTORUN_DEMO:
    if AUTORUN_OLD_DEMO:
      # Let's put some dashed lines between the demos
      number_of_linebreaks = 3
      number_of_dashes = 70
      print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
      print(get_linebreaks_and_dashes(0, number_of_dashes))
      print(get_linebreaks_and_dashes(0, number_of_dashes))
      print(get_linebreaks_and_dashes(number_of_linebreaks, 0))
    demo()
  if AUTORUN_DEBUG_DEMO:
    if AUTORUN_DEMO or AUTORUN_OLD_DEMO:
      # Let's put some dashed lines between the demos
      number_of_linebreaks = 3
      number_of_dashes = 70
      print(get_linebreaks_and_dashes(number_of_linebreaks, number_of_dashes))
      print(get_linebreaks_and_dashes(0, number_of_dashes))
      print(get_linebreaks_and_dashes(0, number_of_dashes))
      print(get_linebreaks_and_dashes(number_of_linebreaks, 0))
    debug_demo()

######################################################################
# END OF FILE
######################################################################
