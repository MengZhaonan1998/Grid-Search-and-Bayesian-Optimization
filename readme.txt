--------------------------------------------------------------Readme----------------------------------------------------------------------
Welcome to our second project of <linear algebra and optimization for machine learning>!
This project is contributed by Zuhair, Alvedian Mauditra Aulia Ma, and Zhaonan Meng.

-----------------------------------------------------------------------
Under the directory 'project 2', you can find following files:     
1. GridSearch.py 					     
2. start_bayesian_opt.py				     
3. GD_BO.py					     
4. MBFGS_BO.py					     
5. optimization.py					     
6. heart.csv			
-----------------------------------------------------------------------		     
-----------------------------------------------------------------------

To access the results of the first part, please directly run GridSearch.py:

	The file GridSearch.py is a python script for the grid search (GS) part of the project.
	You can run the python file and input or choose the given options.
	You need to specify (choose) the number of grid points, the grid type, the scale type, and the method of processing.
	The outputs are the best performance and the corresponding C and gamma, 
	computational time and a 3D plot of the five-fold cross-validation accuracy (performance).

	Based on our experiment and the result of the grid search, we choose the following parameters:
	number of grid points for each parameter: 100 (total 100x100 grid points) (input 100)
	grid type: nonuniform (option 2)
	scale type: standard (option 1)
	method of processing: multiprocessing (option 1)
	On our computer, the computational time is about 3.5 minutes.

------------------------------------------------------------------------

To access the results of the second part, please directly run start_bayesian_opt.py:

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


