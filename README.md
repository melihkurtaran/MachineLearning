# Machine Learning Projects

## Supervised learning 
It is a type of machine learning where the algorithm is trained using labeled data. The goal is to learn a function that can predict the output given a new input. Common supervised learning tasks include classification and regression.

- The project uses a dataset collected using readings from a multi-spectral imaging sensor mounted on a drone, to develop supervised machine learning models. The dataset is loaded and analyzed using various libraries such as Pandas, Numpy, SciPy, scikit-learn, and Matplotlib.
- The dataset is loaded from a CSV file on GitHub, and is then cleaned, preprocessed and transformed. Various models are trained and evaluated using methods such as PCA, SVM, Decision Trees, Random Forest, Linear Regression, Voting Classifier, Stacking Classifier, and Polynomial Features. 
- The models are optimized using GridSearchCV, RepeatedKFold, and cross_val_score. Metrics such as accuracy, precision, F1 score, recall and confusion matrix are used to evaluate the performance of the models. The project aims to classify and regression tasks using the dataset.

## Instance-based learning 
It is a type of machine learning where the algorithm learns by storing instances or examples and then generalizing to new instances based on a similarity measure. This is typically used for classification and regression problems.

- The project is using quadratic programming solver and CVXOPT library to solve for the SVM.

- The project is  plotting the samples and displaying the decision function. The samples are plotted using a scatter plot, with class 0 samples represented as green dots and class 1 samples represented as blue crosses. The decision function is represented as a hyperplane, which separates the two classes. The project also display the solution of SVM with quadratic programming solver and the decision function.
- This project is using a technique called the kernel trick, which maps the data from the original space to a higher dimensional space, where the data may be linearly separable. The kernel trick is done by transforming the data using a function Φ(x) = (x1x2, x1^2 + x2^2). By using this function, the project maps the data from 2-dimensional space to a new 2-dimensional space. Then the project plots the samples in the transformed space, which shows the samples are now linearly separable.
- The project provides a clear demonstration of how the kernel trick can be used to make non-linearly separable data linearly separable in SVMs. It also shows how the data is transformed from the original space to a higher dimensional space by using a function called Φ(x) = (x1x2, x1^2 + x2^2) and how the samples are plotted in the transformed space.

## Unsupervised learning
It is a type of machine learning where the algorithm is trained using unlabeled data. The goal is to find patterns or structure in the data. Common unsupervised learning tasks include clustering and dimensionality reduction.


## FFNN (Feedforward Neural Network)
It is a type of neural network where the information flows in one direction from input to output. It is commonly used for tasks such as image recognition, speech recognition, and natural language processing.

## RBFNN (Radial Basis Function Neural Network)
It is a type of neural network that uses radial basis functions as activation functions. It is commonly used for tasks such as function approximation and time series prediction.

## Self-organizing map (SOM) 
It is a type of unsupervised learning algorithm that is used to project high-dimensional data onto a low-dimensional space while preserving the topological structure of the data. SOMs are used for visualization and dimensionality reduction.
