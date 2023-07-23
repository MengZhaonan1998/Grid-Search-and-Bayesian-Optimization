# Grid Search and Bayesian Optimization

This project is contributed by Zhaonan Meng, Alvedian Mauditra Aulia Matin, and Zuhair

## Description

This project is the homework of the TU Delft course 'Linear Algebra and Optimization for Machine Learning.' In this project, we studied two methods for hyperparameter tuning: grid search and Bayesian optimization. We implemented both approaches and applied them to tune the hyperparameters (kernel size and regularization) of a support vector machine. Further outcomes of the project can be found in the PDF file 'project2_BayesianOptSVM.pdf' in this repository.

## Getting Started

### Hierarchy
* GridSearch.py
* start_bayesian_opt.py
* GD_BO.py
* MBFGS_BO.py
* optimization.py
* heart.csv

### Dependencies

To run the program you need to make sure that you have installed the Python package below:
* os
* sys
* tqdm
* multiprocessing 
* numpy
* scipy
* pandas
* sklearn
* matplotlib
* time
* csv

### Executing program

#### To access the results of the first part (grid search), please directly run GridSearch.py:
The file GridSearch.py is a python script for the grid search (GS) part of the project. You can run the python file and input or choose the given options. You need to specify (choose) the number of grid points, the grid type, the scale type, and the method of processing. The outputs are the best performance and the corresponding C and gamma, computational time and a 3D plot of the five-fold cross-validation accuracy (performance).

	Based on our experiment and the result of the grid search, we choose the following parameters:
	number of grid points for each parameter: 100 (total 100x100 grid points) (input 100)
	grid type: nonuniform (option 2)
	scale type: standard (option 1)
	method of processing: multiprocessing (option 1)
	On our computer, the computational time is about 3.5 minutes.

#### To access the results of the second part (Bayesian optimization), please directly run start_bayesian_opt.py:
The file start_bayesian_opt.py is a python script for the Bayesian optimization (BO) part of the project.
	
 	start_bayesian_opt.py: 
	an interface which displays all results reported.
	start_bayesian_opt.py depends on GD_BO.py, MBFGS_BO.py, optimization.py and heart.csv. So please make sure all files are under the same directory.		
	You can input 1, 2, 3, or 4 to access results of BO equipped with different acquisition functions and optimization algorithms.
	-----1: BO with the first acquisition function and gradient descent method
	-----2: BO with the first acquisition function and Quasi-Newton BFGS method
	-----3: BO with the second acquisition function and gradient descent method
	-----4: BO with the second acquisition function and Quasi-Newton BFGS method
	Please note: As we are a team of three, we mainly focus on the first acquisition function. The second type has not been deeply investigated. 
		     Running BO with the second acquisition function (3 and 4) consumes a large amount of time. 

	GD_BO.py:
	consists of functions of implementing Bayesian optimization utilizing gradient descent method to minimize the acquisition function 	

	MBFGS_BO.py:
	consists of functions of implementing Bayesian optimization utilizing BFGS method to minimize the acquisition function

	optimization.py:
	A toolkit containing different optimization algorithms such as GD and BFGS	
