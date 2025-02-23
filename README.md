# California Housing Price Prediction Project

## Background
The California Housing Price Prediction project aims to predict housing prices in
California using multiple machine learning models and feature engineering techniques.
The dataset used contains 20,640 observations of 14 variables, including
Median_House_Value, Median_Income, Median_Age, Tot_Rooms, Tot_Bedrooms,
Population, Households, Latitude, Longitude, and several distance variables (e.g.,
distance to major cities).

## Data Description
A histogram of Median_House_Value showed that most values fall between $100,000
and $200,000, with a notable spike at $500,000, indicating a potential cap in the data.
The correlation matrix revealed strong correlations between several variables, such as
Tot_Rooms and Tot_Bedrooms (correlation of 0.93), and distance variables with each
other (correlations above 0.9).

![Screenshot 2025-02-23 at 12 22 54 AM](https://github.com/user-attachments/assets/97524914-8692-49c7-8ef6-2dabc7c4c1e1)

# Methods and Result & Interpretation
## Data Preprocessing
The initial dataset was scaled and sampled down to 3,000 observations for ease of
model training. Median house values were scaled down by dividing by 10,000. Several
features, including Tot_Rooms, Population, Median_Income, and other distance-related
variables, showed high skewness and were log-transformed to normalize distributions.
Highly correlated features were identified, but removing them did not lead to significant
improvements in model performance, so they were retained.

![Screenshot 2025-02-23 at 12 24 03 AM](https://github.com/user-attachments/assets/4ba8662d-ca5e-4f1d-aefd-ad64766d07b6)


# Modeling Approaches
Nine different models were implemented and evaluated:
1) Linear Regression: Explained 67.2% of the variance with an RMSE of 6.94.


2) Partial Least Squares (PLS): Used 8 components, explaining 64.5% of the variance
with an RMSE of 6.96.

3) Lasso Regression: Achieved similar performance to linear regression with an RMSE
of 6.94.

![image](https://github.com/user-attachments/assets/10fe8f01-1f1d-451d-9396-15eb8e14c2c1)


5) MARS (Multivariate Adaptive Regression Splines): Showed better performance
with an RMSE of 5.71 and explained 76.3% of the variance.

![image](https://github.com/user-attachments/assets/df88c370-e3c9-4b6c-8284-92b2d7c8338a)


5) Neural Network: Used a network with hidden layers (2 and 3 neurons), achieving an
RMSE of 6.14 and R² of 0.709.

![image](https://github.com/user-attachments/assets/622de024-3005-4df9-a4ca-53e4c981a94e)

6) K-Nearest Neighbors (KNN): Produced an RMSE of 6.75 and an R² of 0.677.

7) Support Vector Machine (SVM): Achieved an RMSE of 5.70, explaining 76.9% of
the variance.

8) CART (Classification and Regression Tree): Had the lowest R² value (0.572) with
an RMSE of 7.45, indicating less effective prediction.

![image](https://github.com/user-attachments/assets/cc1e103d-0c0a-413d-8427-c2f31b9ce00b)

9) Random Forest: Provided the best overall performance with an RMSE of 5.52 and
an R² of 0.768.

![image](https://github.com/user-attachments/assets/eac52993-801f-40f0-b5eb-21de1c7600c0)


# Model Comparison
The models were evaluated based on RMSE and R² values. Random Forest performed
the best, with Random Forest slightly outperforming others in terms of RMSE (5.52) and
variance explained (76.8%). MARS and SVM also showed strong performance,
suggesting they capture key nonlinear relationships better than other approaches.

![image](https://github.com/user-attachments/assets/a401bd2b-d043-4d7a-a5b2-8a8eb0076f17)

# Interpretation and Insights
The distance to the coast was identified as the most significant factor influencing house
prices, with proximity to the coast resulting in higher values. Other important factors
included Median_Income and Latitude. The Random Forest model highlighted feature
importance and offered a flexible approach to capture interactions among predictors.
The comparison between models shows the advantage of using ensemble and non-
linear methods in capturing the complex patterns in housing prices. Despite the
relatively good overlap between actual and predicted values, deviations suggest issues
in capturing extreme values, likely due to non-linearity in the data.

# Conclusion
This project explored various regression models to predict housing prices in California.
The Random Forest model emerged as the best-performing approach, capturing key
features affecting house prices. Future work could include fine-tuning the models
further, incorporating additional data (e.g., neighborhood quality), and exploring more
advanced deep learning methods such as convolutional neural networks (CNNs) and
recurrent neural networks (RNNs), which could help capture spatial and sequential
patterns in the data respectively, improving prediction accuracy.
