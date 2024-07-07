This project focuses on distributed machine learning model training using PyTorch and Ray, a framework for building and running distributed applications.
Explanation
1.	Environment Setup:
•	environment_setup.sh sets up the necessary environment by installing Python, Miniconda, and the required Python packages including pytorch and ray.
2.	Distributed Data Processing and Model Training:
•	distributed_training.py script performs distributed training of a simple convolutional neural network (CNN) using Ray and PyTorch.
•	The script initializes Ray for distributed computing and defines a simple CNN model.
•	The train_cnn function handles the model training process, including loading the MNIST dataset, initializing the model, defining the loss function and optimizer, and running the training loop.
•	The script also includes a hyperparameter tuning section using Ray Tune, specifying a search space for learning rate and batch size, and utilizing the ASHA scheduler for efficient hyperparameter optimization.
•	The training results are reported using Ray Tune's CLIReporter.

