# Breast-Cancer-Diagnosis-Classifier
 This project is focused on building a decision tree classifier to diagnose breast cancer based on features extracted from breast biopsies. The decision tree classifier aims to classify breast biopsies as either malignant  or benign based on various features.

1. Dataset: We load the breast cancer dataset using `datasets.load_breast_cancer()`. This dataset contains features extracted from breast biopsy samples, and we use this data to train and test our classifier.

2. Data Splitting: The dataset is split into training and testing sets using `train_test_split()`. We reserve 40% of the data for testing, ensuring that the classes are stratified for an unbiased evaluation.

3. Decision Tree Classifier: We create a decision tree classifier using the `tree.DecisionTreeClassifier` from scikit-learn. The classifier is trained on the training data with a specified criterion ('entropy') and a minimum number of samples required to split an internal node ('min_samples_split') set to 6. The classifier's accuracy on the test data is also calculated and printed.

4. Tree Visualization: We visualize the decision tree using `tree.plot_tree()` to provide a clear representation of how the classifier is making decisions.

5. Tree Depth Experimentation: We experiment with different depths for the decision tree to understand how it affects the model's performance. We train multiple decision trees with varying depths and plot the training and testing accuracies against the depth of the tree.

6. Hyperparameter Tuning: We perform hyperparameter tuning using Grid Search (`GridSearchCV`) to find the optimal maximum depth for the decision tree. The best parameters for the model are printed, and a decision tree with the optimized depth is visualized.

Instructions

1. Make sure you have Python installed on your system along with the necessary libraries, including scikit-learn and matplotlib.

2. Copy and paste the provided code into a Python environment or script.

3. Run the code to load the dataset, split it into training and testing sets, and train a decision tree classifier. The classifier's accuracy is printed.

4. You can also experiment with different tree depths by plotting the training and testing accuracies for varying depths.

5. Perform hyperparameter tuning to find the optimal maximum tree depth by running the Grid Search part of the code.

6. Observe the decision tree visualizations to understand how the classifier is making decisions.

