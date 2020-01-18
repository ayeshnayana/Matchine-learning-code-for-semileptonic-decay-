# Matchine-learning-code-for-semileptonic-decay-
There are three parts to this project: 

1. Creating Pseudo-data based on experimental data points (python or C++)
2. Training a set of Artificial Neural Networks (ANN) on said Pseudo-data (PD)
        and having it output a discrete set of points at a number of q^2 values you wish
        with a mean and a standard deviation for the target (python)
3. Taking said averaged ANN calculation and running it to find a bound for F_+(0) and V_cd
        (this step is specific to our decay and not very modular, but could use as a base) (Mathematica)
        


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
STEP 2: TRAINING ANN AND THE ANN'S FINAL CALCULATION	

	a. Seperate the pseudo data (pd) to smaller batches and obtain the training and target set using dataMonteCarlo.py:
		-File inputs: 
			-First input is the pd prepared in the step 1 are read into dataMonteCarlo.py in csv format.
			-Next define the number of data points per batch 
		-Output:
			-Traning data set, which is x_train = data_points_per_batch*2*number_of_batches 3d matrix:.
			-Target set, which is y_train = data_points_per_batch*number _of_batches 2d matrix
		-The training data and target data are seved into dataMonteCarlo.mat for record keeping

	b. Artificial neural network training (ANN):
		-Inputs: 
			-Training set data: x_train
			-Target set data: y_train
			-input_layer_size :The input layer is 2 dimensional array with q^2 and q^2_{normalized} data.
			-hidden_layer1_size: Define the number of first hidden layer nodes
			-hidden_layer2_size: Define the number of second hidden layer nodes
			-num_labels: This is the output layer size. By default the value is 1. The num_labels can be changed 
					depending on the desired number of output. However, the number of outputs should be 
					compatible with the size of the target data set. 
			-maxiter: The number of iterations to minimize the error function using non-linear conjugate gradient method(NLCG).
					By default this value is set to 50 iterations. More iteratiopns leads to accurate ANN fit.
			-lambda_reg: The ANN is compatible with regularization to overcome the possibility of over fit. Howvere, by
					default the regularization is turned off. 


		-Outputs:
			-Trained first hidden layer nodes: TTheta1
			-Trained second hidden layer nodes: TTheta2
			-Trained output layer nodes: TTheta3
		-The output is saved into run_data.mat file for record keeping.

		-Notes on ANN training:
			-The backpropagation and the error function is implemented in nnCostfunction.py. 
			-The checkNNGradients.py is included as a safty check. This will calculate analytical gradient of the error function
				for a sample parameter set. The numerical gradients obtained by the backpropagation algorithm then compared 
				with the analytical gradients. If the difference between two gradients is less than 1e-10 then the 
				backpropagation is correctly implemented. 
			-To minimize the error function we use scipy optimize.minimize open source minimization algorithm. This can be 
				found at https://github.com/scipy/scipy/blob/master/scipy/optimize/lbfgsb.py. This algorithm is based on NLCG.
			
		-System requirements:
			-This software is compatible with Python 3.7.4. Before use please install scipy, nupy,csv, pandas, io and matplotlib
				
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
