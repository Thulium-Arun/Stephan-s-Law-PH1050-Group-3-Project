# Stephan-s-Law-PH1050-Group-3-Project
As part of our coursework, we are mandated to write this code and submit on github. This is the result of that.

#Split up of Work:

- Arun Subramanian (EP26B005): Simulated Data Generation, (1.3) Numerical Root Finding, Report Writing, Assignment of Work, Project Supervision.
- Baavesh MV (EP26B007): (1.2) Convergence Study
- Mishka Singh (EP26B017): (2.1) Data and Statistics
- Shreyansh (EP26B029): (3) Polynomial Fitting
- Tharun Karthick (EP26B036): Real Data Collection
- Ekarshi (PH26B003): (1.1) Euler and RK4 Steps 
- Jayadeep (CE24B073): (2.2) Data and Statistics

#How to Run Code:

### 0.Data Generation.py

`python "0.Data Generation.py`

Dependencies: 
- Pandas
- Numpy
- Matplotlib
- Scipy 

Has parameters (in the code) and generates noisy temperature versus time values. With fan, without fan, and using stephan's law. 

Output: cooling_datasets.csv 

### 1.1 Euler and RK4.py

`python "1.1 Euler and RK4.py"`

Dependencies:
- Numpy
- Matplotlib

Gives the graph of Euler and RK4 Simulation vs analytic solution

Output: Euler,RK4 Plot.png

### 1.2 Error Analysis.py

`python "1.2 Error Analysis.py`

Dependencies:
- Numpy
- Matplotlib

Gives the graph of Euler and RK4 error from analytic solution vs the step size. 

Output: Error Step Size Analysis.png

### 1.3 Bisection Numerical Root Finder.py

`python "1.3 Bisection Numerical Root Finder.py"`

Dependencies:
- Numpy

Output: Time taken to go from one temperature to another (parameters in code) with and without fan. 

### 2 Data and Statistics.py

`python "2 Data and Statistics.py" cooling_data.csv`

Dependencies:
- Csv
- Sys
- MatplotLib
- Numpy
- Scipy

Output: Q-Q Graphs for each data set in cooling_datasets.csv, Residual Histogram and Scatterplot. 


### "3 Polynomial Fitting.py"

`python "3 Polynomial Fitting.py"`

Dependencies:
- Matplotlib
- Numpy
- Scipy
- Pandas

Output:





