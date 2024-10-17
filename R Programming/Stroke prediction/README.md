# Stroke Prediction using Machine Learning Models

## Introduction
Stroke is one of the leading causes of mortality and disability worldwide. According to the Global Stroke Factsheet published in 2022, the lifetime risk of having a stroke has increased by 50% in the last 17 years, with 1 in 4 people over the age of 25 likely to experience a stroke. Each year, approximately 12.2 million individuals suffer their first stroke, and 6.5 million of them will die. The global number of stroke survivors exceeds 110 million, and stroke incidence has risen by 70% since 1990.

Stroke is especially prevalent in low- and middle-income nations, where 86% of stroke-related deaths and 89% of disability-adjusted life years are recorded. The World Stroke Organization identifies high systolic blood pressure, high body mass index (BMI), elevated fasting plasma glucose levels, and low glomerular filtration rate as some of the critical metabolic risk factors associated with stroke. Smoking, unhealthy eating, lack of physical activity, environmental risks such as air pollution, and lead exposure further contribute to stroke prevalence.

This project explores various attributes leading to stroke, developing predictive models to help individuals assess their risk of stroke based on simple data inputs.

![3d-medical-figure-with-brain-highlighted_1048-8345](https://github.com/user-attachments/assets/3c3d55d5-c9e9-451f-bf6c-9fcd88ebf2d1)

## Aim and Objectives

Predict the likelihood of stroke risk based on user-inputted data.

Provide an interface that allows anyone to make stroke predictions.

Identify the key indicators associated with stroke risk.

Apply appropriate data mining techniques to the dataset.

Implement and compare suitable machine learning models.

## Model Comparison
To predict stroke, we evaluated several machine learning models, including Random Forest, LightGBM, K-Nearest Neighbors (KNN), Decision Tree, Logistic Regression, and Support Vector Machine (SVM). Each model was assessed based on various performance metrics such as accuracy, precision, recall, and F1 score.

![accuracy comparison of models](https://github.com/user-attachments/assets/de2a1c07-7975-4a9e-8411-22e8b60e830a)
![mae comparison of models](https://github.com/user-attachments/assets/ffd402f0-bf74-4c9b-b181-29bef7554e39)
![mse comparison of models](https://github.com/user-attachments/assets/63a64754-a9a6-4e42-b019-8c733e2b555d)
![f1 score comparison of models](https://github.com/user-attachments/assets/f970d3c7-8a97-460b-985c-e06e66ddeb90)
![recall comparison of models](https://github.com/user-attachments/assets/fb609797-9988-415d-bcff-e7cfc8bdba76)
![precision comparison of models](https://github.com/user-attachments/assets/5a26575b-7953-4657-8b18-89e3f1729215)
![rmse comparison of models](https://github.com/user-attachments/assets/be955f75-c2bf-4376-97cc-de4a5384b2ed)

After evaluating these models, the Random Forest model emerged as the most suitable and accurate for stroke prediction.

## Shiny Framework

I implemented a Shiny framework to create an interactive user interface. This application allows users to input key health metrics and receive an immediate prediction of their stroke risk. The application is designed to be user-friendly and accessible, providing a simple way for individuals to assess their stroke likelihood.

![Screenshot 2024-10-17 at 04 23 46](https://github.com/user-attachments/assets/83fe0b29-6ed7-4fbc-a8c9-415434e036b0)

The framework is structured so users can input data such as age, BMI, glucose levels, heart disease status, and more, to receive predictions in real-time.

## Conclusion

In conclusion, after processing the data and evaluating multiple machine learning models, the Random Forest model proved to be the most accurate. The most important predictors of stroke include age, marital status, heart disease, BMI, and average glucose levels. By incorporating the Shiny framework, I have developed a user-friendly platform that allows individuals to assess their stroke risk using commonly known health indicators.


