import numpy as np
from scipy.interpolate import interp1d
from operator import itemgetter

class ResponseFunction:
	
	
	def __init__(self, fName):
		'''
		Name: The Name of the data file
		x : Data array containing (Energy values directly from diagonalized Hamilotnian)
		y : Data array containing the value |<f|O|0>|^2 which are un-normalized 
		'''
		
		x,xif, y = np.loadtxt(fName, skiprows=1,usecols=(1,2,3), unpack=True)
		
		self.E0 = x[0]-xif[0]
		self.x = x
		self.y = y
		self.Nquad = len(x)
		
		assert len(x)==len(y), 'Error: Vectors should be the same length'
		
		# We must remove any Ek=0 points.
		count = 0
		
		for i in range(0,self.Nquad):
			if(x[i]==0.0):
				count=count+1
		
		self.x = self.x[count:self.Nquad]
		self.y = self.y[count:self.Nquad]
		
		self.Nquad = len(self.x)
		
		# Now we spline the points {k,x_k}
		self.k = np.asarray(range(0,self.Nquad)) 
		self.x_eta = interp1d(self.k, self.x,kind='linear')
		
	def df(self,func,x,eps=1e-10):
		'''
		Compute the Derivative of a function
		'''
		d = func(x+eps)-func(x)
		d = d/eps
				
		return d
	
	
	def weights(self):
		'''
		Function which computes all of the weights according to the notes
		
		returns a vector of lenght w[0:Nquad-1]
		'''
		
		# Define a Local Variable
		x_eta = self.x_eta		
		
		# Compute the Derivative to Obtain the Weights 
		w = np.asarray([ abs(self.df(x_eta,k)) for k in range(0,self.Nquad-1)])
		
		# Store the weights as part of the Response function object
		self.w = w
		
		return w
		
	def scattering_Response(self):
		'''
		This function returns the value of the matrix elements with the L2 Weight factor removed
		
		Length of the vector R, is [0:Nquad-1]
		'''
		
		w = self.weights()
		
		x = self.x[0:(self.Nquad-1)]
		R = np.asarray([ self.y[k]/w[k] for k in range(0,self.Nquad-1)])
		
		return x,R
		
	def combine_Response_functions(R_object_vec):
		'''
		This function will combine all Response functions together
		
		input: Vector with all Response function objects
		output: Energy and Response function values ordered in ascending order  
		'''
		
		N_objects = len(R_object_vec)
		
		print('Number of objects: ', N_objects)
		
		# The Zeroth object
		Oi = R_object_vec[0] # Retrieve the ith object
		Nquad_Oi = Oi.Nquad
		E_Oi,R_Oi = Oi.scattering_Response()
		z = np.asarray([[E_Oi[j],R_Oi[j]] for j in range(0,(Nquad_Oi-1)) ])
		
		# Combine all Response Function objects into a single array
		for i in range(1,N_objects):
			
			Oi = R_object_vec[i] # Retrieve the ith object
			Nquad_Oi = Oi.Nquad
			E_Oi,R_Oi = Oi.scattering_Response()
			
			zi = np.asarray([[E_Oi[j],R_Oi[j]] for j in range(0,(Nquad_Oi-1)) ])
			
			z=np.concatenate((z,zi),axis=0)
		
		# Sort the Array accroding to the Energy
		z=z[np.argsort(z[:, 0])]
		
		# The Combined Length of the new arrays
		new_Len = len(z)
		
		
		E = np.asarray([z[k][0] for k in range(0,new_Len) ])
		R = np.asarray([z[k][1] for k in range(0,new_Len) ])
		
		return E,R
		
		
		
		
