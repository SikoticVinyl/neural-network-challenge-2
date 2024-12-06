# HR Attrition and Department Prediction Model

## Description
A neural network model designed to predict employee attrition and optimal department placement using HR data. This project focuses on implementing a branched neural network architecture to handle multiple prediction tasks simultaneously.

### Key Learning Challenges
- Understanding the implementation of shared layers using the 'x' variable approach versus explicit layer naming
- Grasping the differences between Sequential and Functional API models
- Implementing proper normalization and dropout layers
- Managing multiple outputs in a single neural network

## Architectural Insights

### Shared Layers vs Sequential
One of the main challenges was understanding why we couldn't use a Sequential model. Key learnings:
- Sequential models only support single input → single output
- Functional API needed for multiple outputs (department and attrition)
- Shared layers allow feature learning across both predictions

### Normalization and Regularization
Critical learning points:
- Difference between data scaling (preprocessing) and BatchNormalization (in-model)
- Understanding dropout rates and their impact
- Why regularization matters for HR data

### Model Architecture Challenges
1. Shared Layer Implementation:
```python
# Modern approach using 'x'
x = layers.Dense(64, activation='relu')(input_layer)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
```
vs traditional approach:
```python
# Traditional explicit naming
shared_layer1 = layers.Dense(64, activation='relu')(input_layer)
shared_layer2 = layers.Dense(32, activation='relu')(shared_layer1)
```

## Installation
Required libraries:
- TensorFlow
- pandas
- numpy
- scikit-learn

```bash
pip install tensorflow pandas numpy scikit-learn
```

## Usage
The notebook follows this workflow:
1. Data preprocessing
2. Feature engineering
3. Model creation with branched architecture
4. Training and evaluation

## Future Improvements
1. Feature Engineering
   - Add interaction terms
   - Create derived features
   
2. Model Architecture
   - Experiment with different branch sizes
   - Try residual connections
   
3. Training Process
   - Implement cross-validation
   - Add learning rate scheduling
