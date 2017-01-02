"""Python script for Exercise set 6 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb
from numpy import linalg as LA


class UnsupervisedLearning(object):
    
    '''
    Input:
    iter: iterrations  count for convergence in the SOM 
    algorithm
    
    gaussChangeOverTime: hold the Gaussian values during iteration
    
    eta : Learning rate 
    
    centreDiff: difference between previous and next centre 
    iterations important for convergence 
    
    dataCentreDiff: difference between centre and data during the iteration
    important for convergence too.
    '''
    
    def __init__(self, itera=0,gaussChangeOverTime=[],centreDiff=[],dataCentreDiff=[],eta=0.0):
        self.itera = itera
        self.gaussChangeOverTime= gaussChangeOverTime
        self.centreDiff = centreDiff;
        self.dataCentreDiff = dataCentreDiff;
        self.eta = eta;
        
        
    
    
    def kohonen(self,name,size_k,sigma,tmax):
        """Example for using create_data, plot_data and som_step.
        """
        
        plb.close('all')
        
        eta=self.eta
        
        dim = 28*28
        data_range = 255.0

        # load in data and labels    
        data = np.array(np.loadtxt('data.txt'))
        labels = np.loadtxt('labels.txt')
        
        

        # use name2digits function and print the returned digits  
        
        targetdigits = self.name2digits(name) # assign the four digits that should be used
        print("Digits representation of 'Olagoke Lukman Olabisi' using name2digits Function is: ")
        print(targetdigits) # output the digits that were selected
        
        

        # this selects all data vectors that corresponds to one of the four digits
        data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
        print("Shape of data vectors that corresponds to one of the four digits returned: ")
        print(data.shape)
        dy, dx = data.shape


        
        #size_k = 6 . print size of map
        print("Size of Kohonen Map :" + str (size_k))
        
        

        #sigma = 3.0  print standard deviation
        print("Standard Deviation: " + str(sigma))   
        
        

        #initialise the centers randomly
        centers = np.random.rand(size_k**2, dim) * data_range
         
            
            
        #build a neighborhood matrix
        neighbor = np.arange(size_k**2).reshape((size_k, size_k))
        print("Neighborhood Matrix: " + str(neighbor.shape)) 
        
        

        #print the learning rate
       
        print("Learning rate:" + str(eta))      
       
        #set the random order in which the datapoints should be presented
        i_random = np.arange(tmax) % dy
        np.random.shuffle(i_random)
        
        # use flag to stop loop if true. cross check with the SOM call
        flag = False

        for t, i in enumerate(i_random):
            if (flag == False):
                flag=self.som_step(centers, data[i,:],neighbor,eta,sigma)
                self.itera = t;
            else:
                break    


        # for visualization, you can use this:

        for i in range(size_k**2):
            plb.subplot(size_k,size_k,i+1)
            plb.imshow(np.reshape(centers[i,:], [28, 28]),interpolation='bilinear')
            plb.axis('off')

        # leave the window open at the end of the loop
        plb.show()
        plb.draw()        
        return self.gaussChangeOverTime,self.itera,self.dataCentreDiff,self.centreDiff



    def som_step(self,centers,data,neighbor,eta,sigma):
        """Performs one step of the sequential learning for a 
        self-organized map (SOM).

          centers = som_step(centers,data,neighbor,eta,sigma)

          Input and output arguments: 
           centers  (matrix) cluster centres. Have to be in format:
                             center X dimension
           data     (vector) the actually presented datapoint to be presented in
                             this timestep
           neighbor (matrix) the coordinates of the centers in the desired
                             neighborhood.
           eta      (scalar) a learning rate
           sigma    (scalar) the width of the gaussian neighborhood function.
                             Effectively describing the width of the neighborhood
        """

        size_k = int(np.sqrt(len(centers)))

        #find the best matching unit via the minimal distance to the datapoint
        b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

        # find coordinates of the winner
        a,b = np.nonzero(neighbor == b)
        count =0;
        dataCentreDiff_copy=0;
        dataCentreDiff_Previous=0;
        
        #criteria to stop loop initialised to 1
        difference =1;
        
        # Use flag to stop loop and inform the kohonen function above
        flag =False
        
        
        # update all units
        for j in range(size_k**2):
            
            if (difference>=0.0001):
                #winner function
                count=count +1
                
                # find coordinates of this unit
                a1,b1 = np.nonzero(neighbor==j)

                #count iterations before loop elapses
                

                # calculate the distance and discounting factor
                disc=self.gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])

                # store the value of neighbor hood function
                discVal = disc.copy()
                self.gaussChangeOverTime.append(discVal)

                # update weights        
                centers[j,:] += disc * eta * (data - centers[j,:])

                #change in deltacentre.compute euclid distance between data centre and data
                dataCentreDiff_copy =LA.norm(( centers[j,:] - data), 2)
               
                #Get difference between previous and present norm
                difference = np.absolute(dataCentreDiff_Previous - dataCentreDiff_copy)
                
                #copy the datacentre_copy to previous (temporary storage)
                dataCentreDiff_Previous = dataCentreDiff_copy.copy()
                
                #append difference and datacentre difference to global variable to enable plotting
                self.dataCentreDiff.append(dataCentreDiff_copy)
                self.centreDiff.append(difference)
            else:
                flag = True;
                print("convergence at loop " + str(count) + " of" + " iteration " + str(self.itera) )
                break;
        return flag;
    
    def som_step2(self,centers,data,neighbor,eta,sigma):
        """Performs one step of the sequential learning for a 
        self-organized map (SOM).

          centers = som_step(centers,data,neighbor,eta,sigma)

          Input and output arguments: 
           centers  (matrix) cluster centres. Have to be in format:
                             center X dimension
           data     (vector) the actually presented datapoint to be presented in
                             this timestep
           neighbor (matrix) the coordinates of the centers in the desired
                             neighborhood.
           eta      (scalar) a learning rate
           sigma    (scalar) the width of the gaussian neighborhood function.
                             Effectively describing the width of the neighborhood
        """

        size_k = int(np.sqrt(len(centers)))

        #find the best matching unit via the minimal distance to the datapoint
        b = np.argmin(np.sum((centers - np.resize(data, (size_k**2, data.size)))**2,1))

        # find coordinates of the winner
        a,b = np.nonzero(neighbor == b)
        count =0;
        difference =1;
        centerscopy= np.random.rand(784,)
        flag =False;
       
       
        # update all units
        for j in range(size_k**2):
           
            if (difference>=1e-4):
                #winner function
                count=count +1
                
                # find coordinates of this unit
                a1,b1 = np.nonzero(neighbor==j)

                #count iterations before loop elapses
                

                # calculate the distance and discounting factor
                disc=self.gauss(np.sqrt((a-a1)**2+(b-b1)**2),[0, sigma])
                
                # store the value of neighbor hood function
                discVal = disc.copy()
                self.gaussChangeOverTime.append(discVal)
                
                # update weights        
                centers[j,:] += disc * eta * (data - centers[j,:])
                 
                #change in deltacentre
                
                differences = np.asarray(centers[j,:]- np.asarray(centerscopy))
                
                difference = LA.norm(differences)
             
               
                #Sdifference = np.absolute(dataCentreDiff_Previous - dataCentreDiff_copy )
                
                
                centerscopy = np.empty_like(centers[j,:])
                centerscopy[:] = centers[j,:]                         
                
                
                
                self.centreDiff.append(difference)
            else:
                flag = True;
                print("convergence at loop " + str(count) + " of" + " iteration " + str(self.itera) )
                break;
        
   
        
    def gauss(self,x,p):
        """Return the gauss function N(x), with mean p[0] and std p[1].
        Normalized such that N(x=p[0]) = 1.
        """
        return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

    def name2digits(self,name):
        """ takes a string NAME and converts it into a pseudo-random selection of 4
         digits from 0-9.

         Example:
         name2digits('Felipe Gerhard')
         returns: [0 4 5 7]
         """

        name = name.lower()

        if len(name)>25:
            name = name[0:25]

        primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]

        n = len(name)

        s = 0.0

        for i in range(n):
            s += primenumbers[i]*ord(name[i])*2.0**(i+1)

        import scipy.io.matlab
        Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
        x = Data['x']
        t = np.mod(s,x.shape[0])

        return np.sort(x[t,:])


  

