# Project 2 Grading and Rules

## Rules for Choosing Datasets
- The dataset must have at least 1000 data samples.
- If features are tabular, it must have at least 100 features (this does not apply to image, text, audio).
- Must not be a popular dataset (e.g., MNIST, CIFAR, Iris, Movie Reviews, etc.).
- Must have labels for interpretability / comparison with supervised method.
- At most 1 student can choose a dataset.
- At most 1 student can choose a task (e.g., sentiment analysis, facial expression recognition, etc.).

## Rules for Choosing Methods
- At most 10 students can choose the same model combination.
- At most 20 students can choose a certain model.

## Grading (Project 2)
- a) 1 - granted.
- b) 0.5 - tests with 2 different unsupervised models (0.25 points per model), demonstrated by code.
- c) 0.5 - attempts with 2 different types of features (0.25 per representation), demonstrated by code. Variations on a certain representation (e.g., bag-of-words with and without stopwords) are NOT sufficient.
- d) 0.5 - tuning of parameters (including distance metrics where applicable), grid search, graphs showing the variation of performance depending on the values of the hyperparameters.
- e) 0.5 - comparison with random chance (0.25) and comparison with supervised baseline (0.25).
- f) 6 - questions regarding the knowledge about unsupervised methods.
- g) 0.25 - completeness of the documentation (at least 2 pages, excluding graphs, figures, tables).
- h) 0.5 - interpretation of the results (what do the clusters represent, what are the particularities of the examples in each cluster).
- i) 0.25 - correctness of approach (features used for interpretation are not given at input; train and test separation, i.e. hyperparameters are not set on the test).
- j) 0.5 - 0.5 compensation point may be awarded if there are other additional aspects (testing with the third method, testing in unsupervised transfer learning scenarios, etc.), if the sum from a) to i) is less than 10.

Note: The scores at d), e), g), h), i) are divided by 2 in case of implementing a single unsupervised method.

## Dataset
This dataset provides a comprehensive collection of images aimed at supporting the development of machine learning models for detecting nail diseases, including Acral Lentiginous Melanoma, Healthy Nail, Onychogryphosis, Blue Finger, Clubbing, and Pitting.

### Categories
- `acral_lentiginous_melanoma`
- `healthy_nail`
- `onychogryphosis`
- `blue_finger`
- `clubbing`
- `pitting`

Each subfolder contains images related to the specific nail condition.

### File Format
- JPEG

### Preferred File Formats
- Images are provided in standard image formats like JPEG, which are widely supported across various platforms and tools.

### Use Cases
- Medical image analysis: development of diagnostic tools for early detection of nail diseases.
- Machine learning models: training and evaluation of classification models for nail disease detection.
- Educational purposes: resources for teaching medical students or researchers about nail diseases and their visual characteristics.

## Methods
- Agglomerative Hierarchical Clustering (AHC): bottom-up clustering that merges the closest clusters using a linkage rule (e.g., Ward, average), producing a dendrogram and a final cut for the chosen number of clusters.
- DBSCAN: density-based clustering that groups points in dense regions using `eps` and `min_samples`, finds arbitrary-shaped clusters, and labels sparse points as noise.

## Feature Sets

### Feature Set 1: Color + Simple Intensity Statistics
- Compute on the cropped nail image.
- Color histograms (e.g., 32 bins per channel) in HSV or Lab.
- Mean / std / skewness per channel.
- Rationale: blue finger and lighting/skin tone shifts show up strongly.

### Feature Set 2: HOG (Shape/Edge Structure)
- Resize to a fixed size (e.g., 128x128).
- Compute HOG (cell size / orientations tunable).
- Rationale: nails have edge/shape cues and are more robust to lighting.

### Feature Set 4: Local Binary Pattern (LBP)
- Compute LBP on grayscale images (uniform patterns).
- Build a normalized histogram of LBP codes.
- Rationale: captures local texture changes and surface irregularities.
