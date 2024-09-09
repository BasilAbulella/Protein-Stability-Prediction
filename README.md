  _____                    _                           _             
 |  __ \                  | |                         (_)            
 | |  | | ___  ___ _ __   | |     ___  __ _ _ __ _ __  _ _ __   __ _ 
 | |  | |/ _ \/ _ \ '_ \  | |    / _ \/ _` | '__| '_ \| | '_ \ / _` |
 | |__| |  __/  __/ |_) | | |___|  __/ (_| | |  | | | | | | | | (_| |
 |_____/ \___|\___| .__/  |______\___|\__,_|_|  |_| |_|_|_| |_|\__, |
                  | |                                           __/ |
                  |_|                                          |___/ 
  ____                     _   _____           _       _       
 |  _ \                   | | |  __ \         | |     (_)      
 | |_) | __ _ ___  ___  __| | | |__) | __ ___ | |_ ___ _ _ __  
 |  _ < / _` / __|/ _ \/ _` | |  ___/ '__/ _ \| __/ _ \ | '_ \ 
 | |_) | (_| \__ \  __/ (_| | | |   | | | (_) | ||  __/ | | | |
 |____/ \__,_|___/\___|\__,_| |_|   |_|  \___/ \__\___|_|_| |_|  
                                                                             
   _____ _        _     _ _ _ _           _____              _ _      _   _             
  / ____| |      | |   (_) (_) |         |  __ \            | (_)    | | (_)            
 | (___ | |_ __ _| |__  _| |_| |_ _   _  | |__) | __ ___  __| |_  ___| |_ _  ___  _ __  
  \___ \| __/ _` | '_ \| | | | __| | | | |  ___/ '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \ 
  ____) | || (_| | |_) | | | | |_| |_| | | |   | | |  __/ (_| | | (__| |_| | (_) | | | |
 |_____/ \__\__,_|_.__/|_|_|_|\__|\__, | |_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|
                                   __/ |                                                
                                  |___/                                                                                               

			Deep Learning Based Protein Stability Prediction
					summer, 2024
			
			
made by       : Basil Abulella
Supervised by : Prof. Peter Lackner


table of content:
1. Introduction
2. Project & Folder breakdown
3. How to use
4. Extras

1. Introduction:

Hello everyone who is reading this, I created this file hoping that it would navigate this program easier.
This was my master thesis where i used Keras to try and predict the ΔΔG of mutated proteins.
The goal of this project is to try and replace a part of the MAESTRO program that serves the same function.

2. Project & Folder breakdown

	This project was broken down into multiple files and folders as follows:
	
		project/                        
		├── main_script.py			
		├── readme.txt
		├── requirmenets.txt
		├── modules
		    └── functions
			└── __init__.py
			└── function1.py
			└── function2.py
			└── fucntion3.py
			└── etc....
		    └── input_and_errors.py
		    └── model_training.py
		    └── evaluation.py
		├── files
		    └── SP1.py
		    └── SP2.py
		    └── SP3.py
		    └── SP4.py

		├── data
		    └── csv_data
		    └── npy_data
		    └── data_combination.py
		    └── consensus_data.csv

3.  How to use

	3.1 main_script.py

		this is the main script you should run on bash, where you enter all the desired parameters, 
		please note that some other parameters you have to change manually in the input_and_errors.py file

		To use the program: 
		A) go to the location of the main_script.py
		B) Run the script through bash. 
		C) choose your four parameters, these are the: 
		   file name, the target from the dataset, the scaling method and the number of folds.

		Notes:

		- The script should detect automatically the location of the datasets, 
		so you just need to write the file name with its extension. (eg. SP3_features.tsv)

		- Here is an example of the full command: 
		"python3 main_script.py SP1_features.tsv ddGexp power 5"
		  
		- There are other parameters that can be modified but for this you need to change the inputs_and_errors.py script.
		
		- If you plan to do a hyper parameter run (check 3.2), disabling the evaluate_model is recommened, because it will only display
		evaluation data for the last run only.
		
	3.2 inputs_and_errors.py

		- This script runs the first part of the program and shows some error messages.
		This script also contains other parameters that can be modified.
		in the first few lines you can find multiple parameters with some values in them:

			hidden_layers = [3]          
			activations   = ['LeakyReLU']
			dropouts      = [0.1]
			nodes         = [16]
			epochs        = [200]
			batch_sizes   = [32]
			random_states = [42]

		these parameters can be changed to produce different results, 
		Also you can add multiple parameters in one variable, this will make the program do multiple runs, one with each parameter, for example:

			hidden_layers = [2, 3]          
			activations   = ['LeakyReLU']
			dropouts      = [0.1]
			nodes         = [16]
			epochs        = [200]
			batch_sizes   = [32]
			random_states = [42]
			
		with these settings, there will be two complete different runs, one with 2 hidden layers and one with 3 hidden layers.
		please note that the more parameters you add the more time it will take, with 15 different parameters, it could take up to 2 weeks.

		Notes:
		 
		- As mentioned before, it is recommeneded to disable the evaluate_model in the main_script.py if you plan to do 
		a hyperparameter run.
		- If you move the program to another directory, you need to update the following variables:        
		
			file_dir          = ("your_new_directory_here/project/files/")
			npy_saving_dir    = ("your_new_directory_here/project/data/npy_data/")
			csv_saving_dir    = ("your_new_directory_here/project/data/csv_data/")
		- please be careful with the activations, as it is case sensetive.	
		- this script also have the 4 supported scaling methods: "power", "standard", "minmax" & "robust".
		if you want to add more, you have to import and modify this part:
		  
			# Mapping of scaler names to scaler classes
			scalers = {
			    "power"     : PowerTransformer(),
			    "standard"  : StandardScaler(),
			    "minmax"    : MinMaxScaler(),
			    "robust"    : RobustScaler(),
			}

	4. Extras
	
	
		4.1 data_combination.py
		
			- After each normal run is finished, the main program creates an npy file in the "npy_data" folder, 
			the data_combination.py script combines all .npy files in this folder to create a data consensus .csv file
			consisting of one or multiple runs, the csv file should look something like this:
				
				0			1			2			median			std_deviation		target
				1.52832067012787	1.59123122692108	2.02265357971191	1.59123122692108	0.228132087978479	2.4
				0.780354976654053	0.929060697555542	0.879995584487915	0.879995584487915	0.0624397466419379	4.55
				0.248082727193832	0.616847276687622	0.299320995807648	0.299320995807648	0.169037444422673	4.55
				2.80096936225891	2.48141121864319	2.56701898574829	2.56701898574829	0.137307545333912	0.7

			- the csv file is composed of different parameters:
			
				A) the columns 0, 1 and 2 are the predicted ddG of every separate run, for example if you did 5 runs,
				you will end up having numbered columns: 0, 1, 2, 3 and 4. also the datapoints you see is the 
				median for each position across multiple arrays, they are also sorted to match the target.
				
				B) The median of every data point for all runs.
				
				C) the standard deviation between all runs per datapoint.
				
				D) The DDg for the original values.
				
			- If the script crashes it is most likely becuase the combined datasets dont have the same length, for 
			example combining SP1 and SP4 datasets will result in a crash.

		4.1 hyperparameter_tuning_bashlines

			- This is a simple text file for running multiple different settings at the same time on bash.
		  	this is better used with servers or computers with multicore processors.

			- you load it by going to the location of the main_script and writing: "bash hyperparameter_tuning_bashlines".


		4.2 requirements.txt

			- this is a list of all required packages to run this program.
			- You can easily load all packages directly by going to the location of this file, and running this command on bash:
		  	  "pip install -r requirements.txt"
