[51]:  test's l2:0.229455 
[52]:  test's l2:0.229036 
[53]:  test's l2:0.228664 
[54]:  test's l2:0.22835 
[55]:  test's l2:0.227961 
[56]:  test's l2:0.227578 
[57]:  test's l2:0.227205 
[58]:  test's l2:0.226841 
[59]:  test's l2:0.226474 
[60]:  test's l2:0.226166 
[61]:  test's l2:0.225804 
[62]:  test's l2:0.225434 
[63]:  test's l2:0.225072 
[64]:  test's l2:0.224768 
[65]:  test's l2:0.224409 
[66]:  test's l2:0.224107 
[67]:  test's l2:0.223707 
[68]:  test's l2:0.223333 
[69]:  test's l2:0.222975 
[70]:  test's l2:0.22263 
[71]:  test's l2:0.222274 
[72]:  test's l2:0.221938 
[73]:  test's l2:0.221592 
[74]:  test's l2:0.22124 
[75]:  test's l2:0.220843 
[76]:  test's l2:0.220516 
[77]:  test's l2:0.220127 
[78]:  test's l2:0.219792 
[79]:  test's l2:0.219445 
[80]:  test's l2:0.219118 
[81]:  test's l2:0.218774 
[82]:  test's l2:0.218391 
[83]:  test's l2:0.218107 
[84]:  test's l2:0.217787 
[85]:  test's l2:0.217429 
[86]:  test's l2:0.217082 
[87]:  test's l2:0.216767 
[88]:  test's l2:0.216432 
[89]:  test's l2:0.216108 
[90]:  test's l2:0.215773 
[91]:  test's l2:0.215467 
[92]:  test's l2:0.215153 
[93]:  test's l2:0.214829 
[94]:  test's l2:0.214553 
[95]:  test's l2:0.214281 
[96]:  test's l2:0.21397 
[97]:  test's l2:0.213626 
[98]:  test's l2:0.213307 
[99]:  test's l2:0.212982 
[100]:  test's l2:0.212644 
> lgb.get.eval.result(model, "test", "l2")
  [1] 0.2493028 0.2489278 0.2484925 0.2480117 0.2475712 0.2471321 0.2466946 0.2462591
  [9] 0.2458245 0.2454224 0.2449906 0.2445674 0.2441403 0.2437127 0.2432865 0.2428693
 [17] 0.2424460 0.2420329 0.2416125 0.2412022 0.2407840 0.2403261 0.2399761 0.2395624
 [25] 0.2391273 0.2387987 0.2384565 0.2381303 0.2377877 0.2374017 0.2370025 0.2366072
 [33] 0.2362321 0.2358158 0.2353732 0.2349829 0.2346488 0.2343022 0.2339113 0.2335033
 [41] 0.2330967 0.2327055 0.2322850 0.2319782 0.2315958 0.2312018 0.2308789 0.2305602
 [49] 0.2301543 0.2297730 0.2294547 0.2290364 0.2286640 0.2283500 0.2279608 0.2275777
 [57] 0.2272049 0.2268414 0.2264745 0.2261657 0.2258044 0.2254342 0.2250721 0.2247676
 [65] 0.2244092 0.2241072 0.2237066 0.2233331 0.2229745 0.2226298 0.2222744 0.2219380
 [73] 0.2215919 0.2212398 0.2208428 0.2205157 0.2201268 0.2197921 0.2194451 0.2191185
 [81] 0.2187737 0.2183905 0.2181074 0.2177868 0.2174292 0.2170818 0.2167666 0.2164319
 [89] 0.2161083 0.2157733 0.2154674 0.2151532 0.2148289 0.2145530 0.2142809 0.2139704
 [97] 0.2136255 0.2133073 0.2129819 0.2126443
> 
> pred_y <- predict(model, as.matrix(test_x), reshape = T)
Error in predict.lgb.Booster(model, as.matrix(test_x), reshape = T) : 
  'reshape' argument is no longer supported.
> # Prediction without the reshape argument
> pred_y <- predict(model, as.matrix(test_x))
> 
> # install.packages("pROC")
> library(pROC)
> gbm_auc <- roc(test_y, pred_y)
Setting levels: control = 1, case = 2
Setting direction: controls < cases
> gbm_auc$auc
Area under the curve: 0.9346
> plot(gbm_auc, main = "LightGBM Prediction ROC curve", print.auc = T)
> 
> cm_gbm <- confusionMatrix(as.factor(test_y), as.factor(pred_y))
Error in confusionMatrix.default(as.factor(test_y), as.factor(pred_y)) : 
  The data must contain some levels that overlap the reference.
> # install.packages("pROC")
> library(pROC)
> gbm_auc <- roc(test_y, pred_y)
Setting levels: control = 1, case = 2
Setting direction: controls < cases
> gbm_auc$auc
Area under the curve: 0.9346
> plot(gbm_auc, main = "LightGBM Prediction ROC curve", print.auc = T)
> 
> cm_gbm <- confusionMatrix(as.factor(test_y), as.factor(pred_y))
Error in confusionMatrix.default(as.factor(test_y), as.factor(pred_y)) : 
  The data must contain some levels that overlap the reference.
> # Convert predicted probabilities to binary class labels using 0.5 as threshold
> pred_class <- ifelse(pred_y > 0.5, 1, 0)
> 
> # Convert test_y and pred_class to factors if necessary
> test_y <- as.factor(test_y)
> pred_class <- as.factor(pred_class)
> 
> library(caret)
> 
> cm_gbm <- confusionMatrix(pred_class, test_y)
Warning message:
In confusionMatrix.default(pred_class, test_y) :
  Levels are not in the same order for reference and data. Refactoring data to match.
> print(cm_gbm)
Confusion Matrix and Statistics

          Reference
Prediction    1    2
         1 1275 1200
         2    0    0
                                         
               Accuracy : 0.5152         
                 95% CI : (0.4953, 0.535)
    No Information Rate : 0.5152         
    P-Value [Acc > NIR] : 0.5081         
                                         
                  Kappa : 0              
                                         
 Mcnemar's Test P-Value : <2e-16         
                                         
            Sensitivity : 1.0000         
            Specificity : 0.0000         
         Pos Pred Value : 0.5152         
         Neg Pred Value :    NaN         
             Prevalence : 0.5152         
         Detection Rate : 0.5152         
   Detection Prevalence : 1.0000         
      Balanced Accuracy : 0.5000         
                                         
       'Positive' Class : 1              
                                         
> 
> cm <- conf_mat(table(test_y, pred_y))
Error in `conf_mat()`:
! `x` must have equal dimensions. `x` has 431 columns and 2 rows.
Run `rlang::last_trace()` to see where the error occurred.
> # Ensure both predicted and true values have the same levels
> test_y <- factor(test_y, levels = c(1, 2))  # Replace with the appropriate levels
> pred_class <- factor(pred_class, levels = c(1, 2))
> 
> cm_gbm <- confusionMatrix(pred_class, test_y)
> print(cm_gbm)
Confusion Matrix and Statistics

          Reference
Prediction    1    2
         1 1275 1200
         2    0    0
                                         
               Accuracy : 0.5152         
                 95% CI : (0.4953, 0.535)
    No Information Rate : 0.5152         
    P-Value [Acc > NIR] : 0.5081         
                                         
                  Kappa : 0              
                                         
 Mcnemar's Test P-Value : <2e-16         
                                         
            Sensitivity : 1.0000         
            Specificity : 0.0000         
         Pos Pred Value : 0.5152         
         Neg Pred Value :    NaN         
             Prevalence : 0.5152         
         Detection Rate : 0.5152         
   Detection Prevalence : 1.0000         
      Balanced Accuracy : 0.5000         
                                         
       'Positive' Class : 1              
                                         
> 
> library(yardstick)
> 
> # Use a confusion matrix table
> cm_table <- table(test_y, pred_class)
> 
> # Create confusion matrix using yardstick
> cm <- conf_mat(cm_table)
> autoplot(cm, type = "heatmap")
> 
> # Convert probabilities to class labels
> pred_class <- ifelse(pred_y > 0.5, 1, 0)
> 
> # Calculate precision, recall and F1 score
> TN_gbm <- cm_gbm$table[1,1] # True negatives
> FP_gbm <- cm_gbm$table[1,2] # False positives
> FN_gbm <- cm_gbm$table[2,1] # False negatives
> TP_gbm <- cm_gbm$table[2,2] # True positives
> 
> precision_gbm <- TP_gbm / (TP_gbm + FP_gbm)
> recall_gbm <- TP_gbm / (TP_gbm + FN_gbm)
> F1_score_gbm <- 2 * precision_gbm * recall_gbm / (precision_gbm + recall_gbm)
> 
> cat("Precision: ", precision_gbm, "\nRecall: ", recall_gbm, "\nF1 score: ", F1_score_gbm, "\n")
Precision:  0 
Recall:  NaN 
F1 score:  NaN 
> #----------------------------------------------------------------------------------------------
> # Decison tree ====
> # install.packages('rpart')
> # install.packages('rpart.plot')
> library(rpart)
> library(rpart.plot)
Error in library(rpart.plot) : there is no package called ‘rpart.plot’
> install.packages("rpart.plot")
trying URL 'https://cran.rstudio.com/bin/macosx/big-sur-arm64/contrib/4.4/rpart.plot_3.1.2.tgz'
Content type 'application/x-gzip' length 1023391 bytes (999 KB)
==================================================
downloaded 999 KB


The downloaded binary packages are in
	/var/folders/k6/pj6r9pcs1tzctvhx56bktps80000gn/T//RtmppPDuOQ/downloaded_packages
> 
> library(rpart.plot)
> 
> #----------------------------------------------------------------------------------------------
> # Decison tree ====
> # install.packages('rpart')
> # install.packages('rpart.plot')
> library(rpart)
> library(rpart.plot)
> 
> testing<- rpart(stroke ~.,data=strokeTrain,method = 'class') #build decison tree model
> rpart.plot(testing)  #plot decision tree
> predict<-predict(testing,strokeTest,type='class') #prediction
> 
> predicted_probs<-as.numeric(predict)   #output in prob for ROC plot purpose
> predict_value<-predict(testing,strokeTest,type='vector')  
> #output in vector for finding MSE, MAE, RMSE purpose
> 
> table_mat <- table(strokeTest$stroke, predict) #show confusion matrix table
> table_mat
     predict
       Yes   No
  Yes  914  361
  No   103 1097
> 
> accuracy<-sum(diag(table_mat)/sum(table_mat)) # show accurracy
> print(paste('Accuracy for test :', 1- accuracy))
[1] "Accuracy for test : 0.187474747474747"
> 
> cm_dt <- confusionMatrix(table(strokeTest$stroke, predict)) #show confusion matrix table
> cm_dt  #accuracy 
Confusion Matrix and Statistics

     predict
       Yes   No
  Yes  914  361
  No   103 1097
                                          
               Accuracy : 0.8125          
                 95% CI : (0.7966, 0.8277)
    No Information Rate : 0.5891          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6271          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.8987          
            Specificity : 0.7524          
         Pos Pred Value : 0.7169          
         Neg Pred Value : 0.9142          
             Prevalence : 0.4109          
         Detection Rate : 0.3693          
   Detection Prevalence : 0.5152          
      Balanced Accuracy : 0.8256          
                                          
       'Positive' Class : Yes             
                                          
> # calculate·MSE MAE RMSE
> x<- as.numeric(strokeTest$stroke) #change target variable from yes/no to 1/2
> mse_dt <- mean((x - predict_value)^2)
> mae_dt <- mean(abs(x - predict_value))
> rmse_dt <- sqrt(mean((x - predict_value)^2))
> cat("MSE_DT: ", mse_dt, "\nMAE_DT: ", mae_dt, "\nRMSE_DT: ", rmse_dt)
MSE_DT:  0.1874747 
MAE_DT:  0.1874747 
RMSE_DT:  0.4329835
> library(pROC)
> roc_obj=roc(strokeTest$stroke,predicted_probs ) #AUC/ROC score
Setting levels: control = Yes, case = No
Setting direction: controls < cases
> plot(roc_obj, main = "ROC Curve", print.auc = TRUE, legacy.axes = TRUE) #plot ROC
> 
> cm <- conf_mat(table(strokeTest$stroke, predict))
> autoplot(cm ,type='heatmap') +
+     scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")
Scale for fill is already present.
Adding another scale for fill, which will replace the existing scale.
> 
> # Calculate precision, recall and F1 score
> TN_dt <- cm_dt$table[1,1] # True negatives
> FP_dt <- cm_dt$table[1,2] # False positives
> FN_dt <- cm_dt$table[2,1] # False negatives
> TP_dt <- cm_dt$table[2,2] # True positives
> 
> precision_dt <- TP_dt / (TP_dt + FP_dt)
> recall_dt <- TP_dt / (TP_dt + FN_dt)
> F1_score_dt <- 2 * precision_dt * recall_dt / (precision_dt + recall_dt)
> 
> cat("Precision: ", precision_dt, "\nRecall: ", recall_dt, "\nF1 score: ", F1_score_dt, "\n")
Precision:  0.7524005 
Recall:  0.9141667 
F1 score:  0.8254327 
> #----------------------------------------------------------------------------------------------
> # SVM ====
> #install.packages('e1071')
> #install.packages('yardstick')
> library(e1071)
> library(yardstick)
> 
> svmtrain <- svm(formula = stroke ~ age, #build svm model 
+                 data = strokeTrain,
+                 type = 'C-classification',
+                 kernel = 'linear')
> 
> svmpredict <- predict(svmtrain, newdata = strokeTest)#predict 
> svmpredict_value<-as.numeric(svmpredict)
> #convert output to numeric for finding MSE,MAE,RMSE purpose
> 
> cm_svm <- confusionMatrix(table(strokeTest$stroke, svmpredict))
> #use conf_mat from yardstick library to visualize
> cm_svm
Confusion Matrix and Statistics

     svmpredict
      Yes  No
  Yes 899 376
  No  297 903
                                          
               Accuracy : 0.7281          
                 95% CI : (0.7101, 0.7455)
    No Information Rate : 0.5168          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.4567          
                                          
 Mcnemar's Test P-Value : 0.002641        
                                          
            Sensitivity : 0.7517          
            Specificity : 0.7060          
         Pos Pred Value : 0.7051          
         Neg Pred Value : 0.7525          
             Prevalence : 0.4832          
         Detection Rate : 0.3632          
   Detection Prevalence : 0.5152          
      Balanced Accuracy : 0.7288          
                                          
       'Positive' Class : Yes             
                                          
> # visualize accuracy====
> cm <- conf_mat(table(strokeTest$stroke, svmpredict))
> autoplot(cm ,type='heatmap')+
+     scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")
Scale for fill is already present.
Adding another scale for fill, which will replace the existing scale.
> 
> library(pROC)
> roc_obj=roc(strokeTest$stroke,svmpredict_value ) #AUC/ROC score
Setting levels: control = Yes, case = No
Setting direction: controls < cases
> plot(roc_obj, main = "ROC Curve", print.auc = TRUE, legacy.axes = TRUE) #plot ROC 
> 
> x<- as.numeric(strokeTest$stroke)
> mse_svm <- mean((x - svmpredict_value)^2)
> mae_svm<- mean(abs(x - svmpredict_value))
> rmse_svm <- sqrt(mean((x - svmpredict_value)^2))
> cat("MSE_SVM: ", mse_svm, "\nMAE_SVM: ", mae_svm, "\nRMSE_SVM: ", rmse_svm)
MSE_SVM:  0.2719192 
MAE_SVM:  0.2719192 
RMSE_SVM:  0.5214587
> # Calculate precision, recall and F1 score
> TN_svm <- cm_svm$table[1,1] # True negatives
> FP_svm <- cm_svm$table[1,2] # False positives
> FN_svm <- cm_svm$table[2,1] # False negatives
> TP_svm <- cm_svm$table[2,2] # True positives
> 
> precision_svm <- TP_svm / (TP_svm + FP_svm)
> recall_svm <- TP_svm / (TP_svm + FN_svm)
> F1_score_svm <- 2 * precision_svm * recall_svm / (precision_svm + recall_svm)
> 
> cat("Precision: ", precision_svm, "\nRecall: ", recall_svm, "\nF1 score: ", F1_score_svm, "\n")
Precision:  0.7060203 
Recall:  0.7525 
F1 score:  0.7285196 
> str(strokeTrain)
'data.frame':	5742 obs. of  19 variables:
 $ gender                  : num  1 0 0 0 1 0 1 1 0 0 ...
 $ age                     : num  14 25 80 2 59 45 37 64 23 37 ...
 $ hypertension            : num  0 0 0 0 0 0 0 0 0 0 ...
 $ heart_disease           : num  0 0 0 0 0 0 0 0 0 0 ...
 $ ever_married            : num  0 0 1 0 1 0 1 1 0 1 ...
 $ avg_glucose_level       : num  0.00633 0.00184 0.00738 0.00238 0.00939 ...
 $ bmi                     : num  13.9 24.5 22.9 6.7 18.5 25.4 13.9 19.4 21.6 12.8 ...
 $ smoking_status          : num  0 1 1 0 1 0 1 2 2 0 ...
 $ stroke                  : Factor w/ 2 levels "Yes","No": 2 2 2 2 2 2 2 2 2 2 ...
 $ work_typechildren       : num  1 0 0 1 0 0 0 0 0 0 ...
 $ work_typegov_job        : num  0 0 0 0 0 0 0 0 0 0 ...
 $ work_typenever_worked   : num  0 0 0 0 0 0 0 0 0 0 ...
 $ work_typeprivate        : num  0 1 0 0 0 1 1 0 1 1 ...
 $ work_typeself_employed  : num  0 0 1 0 1 0 0 1 0 0 ...
 $ residence_typerural     : num  0 0 1 1 0 1 0 1 1 0 ...
 $ residence_typeurban     : num  1 1 0 0 1 0 1 0 0 1 ...
 $ age_status              : num  2 3 5 0 4 4 4 4 3 4 ...
 $ avg_glucose_level_status: num  2 3 1 3 0 3 1 3 1 0 ...
 $ bmi_status              : num  2 3 3 0 2 3 2 3 3 1 ...
> #-----------------------------------------------------------------------------------------------
> #ruis
> #----------------------------------------------------------------------------
> colnames(strokeTrain)
 [1] "gender"                   "age"                      "hypertension"            
 [4] "heart_disease"            "ever_married"             "avg_glucose_level"       
 [7] "bmi"                      "smoking_status"           "stroke"                  
[10] "work_typechildren"        "work_typegov_job"         "work_typenever_worked"   
[13] "work_typeprivate"         "work_typeself_employed"   "residence_typerural"     
[16] "residence_typeurban"      "age_status"               "avg_glucose_level_status"
[19] "bmi_status"              
> #Encoding to numerical for stroke (Train)
> strokeTrain$stroke = factor(strokeTrain$stroke,
+                             levels = c('No', 'Yes'),
+                             labels = c(0, 1)) 
> #Encoding to numerical for stroke (Test)
> strokeTest$stroke = factor(strokeTest$stroke,
+                            levels = c('No', 'Yes'),
+                            labels = c(0, 1)) 
> 
> # Random Forest====
> # install.packages("randomForest")
> library(randomForest)
> library(e1071)
> library(yardstick)
> #Train the random forest model
> rf_model <- randomForest(stroke ~ ., data = strokeTrain)
> plot(rf_model)
> print(rf_model)

Call:
 randomForest(formula = stroke ~ ., data = strokeTrain) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 4

        OOB estimate of  error rate: 3.12%
Confusion matrix:
     0    1 class.error
0 2767   17 0.006106322
1  162 2796 0.054766734
> varImpPlot(rf_model)
> var_importance <- importance(rf_model)
> print(var_importance)
                         MeanDecreaseGini
gender                       1.212431e+02
age                          5.652162e+02
hypertension                 5.678675e+01
heart_disease                2.009007e+02
ever_married                 8.937633e+01
avg_glucose_level            9.596943e+01
bmi                          1.032800e+02
smoking_status               1.380755e+02
work_typechildren            5.583873e+00
work_typegov_job             2.737053e+01
work_typenever_worked        3.695500e-05
work_typeprivate             2.711915e+02
work_typeself_employed       7.651872e+01
residence_typerural          6.691127e+01
residence_typeurban          6.171693e+01
age_status                   6.582323e+02
avg_glucose_level_status     7.175133e+01
bmi_status                   2.000764e+02
> sorted_importance <- sort(var_importance, decreasing = TRUE)
> print(sorted_importance)
 [1] 6.582323e+02 5.652162e+02 2.711915e+02 2.009007e+02 2.000764e+02 1.380755e+02
 [7] 1.212431e+02 1.032800e+02 9.596943e+01 8.937633e+01 7.651872e+01 7.175133e+01
[13] 6.691127e+01 6.171693e+01 5.678675e+01 2.737053e+01 5.583873e+00 3.695500e-05
> 
> for (i in seq_along(sorted_importance)) {
+     var_name <- names(strokeTrain)[i]
+     importance_value <- sorted_importance[i]
+     print(paste(var_name, ":", importance_value))
+ }
[1] "gender : 658.232302594202"
[1] "age : 565.216247019031"
[1] "hypertension : 271.191478163413"
[1] "heart_disease : 200.900717981667"
[1] "ever_married : 200.076390507254"
[1] "avg_glucose_level : 138.075495099869"
[1] "bmi : 121.24308007412"
[1] "smoking_status : 103.280008345743"
[1] "stroke : 95.9694306153009"
[1] "work_typechildren : 89.3763294046481"
[1] "work_typegov_job : 76.5187162302562"
[1] "work_typenever_worked : 71.7513290479568"
[1] "work_typeprivate : 66.9112728890746"
[1] "work_typeself_employed : 61.7169302166192"
[1] "residence_typerural : 56.7867522477244"
[1] "residence_typeurban : 27.3705288954526"
[1] "age_status : 5.58387256665545"
[1] "avg_glucose_level_status : 3.69550047536507e-05"
> #Make predictions on test set
> rf_preds <- predict(rf_model,strokeTest)
> table_mat <-confusionMatrix(table(strokeTest$stroke, rf_preds))
> table_mat
Confusion Matrix and Statistics

   rf_preds
       0    1
  0 1185   15
  1  318  957
                                          
               Accuracy : 0.8655          
                 95% CI : (0.8514, 0.8787)
    No Information Rate : 0.6073          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.7326          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.7884          
            Specificity : 0.9846          
         Pos Pred Value : 0.9875          
         Neg Pred Value : 0.7506          
             Prevalence : 0.6073          
         Detection Rate : 0.4788          
   Detection Prevalence : 0.4848          
      Balanced Accuracy : 0.8865          
                                          
       'Positive' Class : 0               
                                          
> 
> # Evaluate the performance for model
> # Example of using AUC as a metric
> library(pROC)
> rf_preds <- as.numeric(as.character(rf_preds))
> rf_preds <- ordered(rf_preds, levels = c("0", "1"))
> print(rf_preds)
   [1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  [41] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  [81] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [121] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [161] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [201] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [241] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
 [281] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [321] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [361] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
 [401] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [441] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [481] 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [521] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [561] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [601] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [641] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [681] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
 [721] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [761] 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
 [801] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [841] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [881] 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [921] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
 [961] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 [ reached getOption("max.print") -- omitted 1475 entries ]
Levels: 0 < 1
> 
> table(strokeTest$stroke)

   0    1 
1200 1275 
> 
> rf_auc <- auc(strokeTest$stroke, rf_preds)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
> str(rf_auc)
 'auc' num 0.869
 - attr(*, "partial.auc")= logi FALSE
 - attr(*, "percent")= logi FALSE
 - attr(*, "roc")=List of 15
  ..$ percent           : logi FALSE
  ..$ sensitivities     : num [1:3] 1 0.751 0
  ..$ specificities     : num [1:3] 0 0.988 1
  ..$ thresholds        : num [1:3] -Inf 0.5 Inf
  ..$ direction         : chr "<"
  ..$ cases             : num [1:1275] 0 0 0 0 0 0 0 0 0 0 ...
  ..$ controls          : num [1:1200] 0 0 0 0 0 0 0 0 0 0 ...
  ..$ fun.sesp          :function (thresholds, controls, cases, direction)  
  ..$ auc               : 'auc' num 0.869
  .. ..- attr(*, "partial.auc")= logi FALSE
  .. ..- attr(*, "percent")= logi FALSE
  .. ..- attr(*, "roc")=List of 8
  .. .. ..$ percent      : logi FALSE
  .. .. ..$ sensitivities: num [1:3] 1 0.751 0
  .. .. ..$ specificities: num [1:3] 0 0.988 1
  .. .. ..$ thresholds   : num [1:3] -Inf 0.5 Inf
  .. .. ..$ direction    : chr "<"
  .. .. ..$ cases        : num [1:1275] 0 0 0 0 0 0 0 0 0 0 ...
  .. .. ..$ controls     : num [1:1200] 0 0 0 0 0 0 0 0 0 0 ...
  .. .. ..$ fun.sesp     :function (thresholds, controls, cases, direction)  
  .. .. ..- attr(*, "class")= chr "roc"
  ..$ call              : language roc.default(response = response, predictor = predictor, auc = TRUE)
  ..$ original.predictor: Ord.factor w/ 2 levels "0"<"1": 1 1 1 1 1 1 1 1 1 1 ...
  ..$ original.response : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
  ..$ predictor         : num [1:2475] 0 0 0 0 0 0 0 0 0 0 ...
  ..$ response          : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
  ..$ levels            : chr [1:2] "0" "1"
  ..- attr(*, "class")= chr "roc"
> # Evaluate the accuracy of the random forest model
> # Convert rf_preds to numeric
> rf_preds <- as.numeric(as.character(rf_preds))
> # Evaluate the accuracy of the random forest model
> rf_acc <- sum(rf_preds == strokeTest$stroke) / nrow(strokeTest)
> print(rf_acc)
[1] 0.8654545
> # Calculate MSE, MAE, RMSE ----
> test_rf <- as.integer(strokeTest$stroke)
> mse_rf <- mean((test_rf - rf_preds)^2)
> mae_rf <- mean(abs(test_rf - rf_preds))
> rmse_rf <- sqrt(mean((test_rf - rf_preds)^2))
> cat("MSE_rf: ", mse_rf, "\nMAE_rf: ", mae_rf, "\nRMSE_rf: ", rmse_rf)
MSE_rf:  1.379394 
MAE_rf:  1.122424 
RMSE_rf:  1.174476
> # Create confusion matrix
> cm_rf <- confusionMatrix(as.factor(rf_preds), strokeTest$stroke)
> print(cm_rf)
Confusion Matrix and Statistics

          Reference
Prediction    0    1
         0 1185  318
         1   15  957
                                          
               Accuracy : 0.8655          
                 95% CI : (0.8514, 0.8787)
    No Information Rate : 0.5152          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.7326          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       
                                          
            Sensitivity : 0.9875          
            Specificity : 0.7506          
         Pos Pred Value : 0.7884          
         Neg Pred Value : 0.9846          
             Prevalence : 0.4848          
         Detection Rate : 0.4788          
   Detection Prevalence : 0.6073          
      Balanced Accuracy : 0.8690          
                                          
       'Positive' Class : 0               
                                          
> # Visualize CM and ROC ----
> library(ggplot2)
> library(ROCR)
> # Confusion Matrix plot
> cmv_rf <- conf_mat(table(strokeTest$stroke, rf_preds))
> autoplot(cmv_rf, type = 'heatmap', label = TRUE) +
+     scale_fill_gradient(low = "#F3E5AB", high = "#F1A680") +
+     labs(title = paste("Confusion Matrix\nAccuracy:", round(cm_rf$overall["Accuracy"], 3)),
+          x = "Predicted Class", y = "Actual Class")+
+     theme(plot.title = element_text(hjust = 0.5))
Scale for fill is already present.
Adding another scale for fill, which will replace the existing scale.
> # ROC curve plot
> roc_rf <- roc(strokeTest$stroke, rf_preds, levels = rev(levels(strokeTest$stroke)))
Setting direction: controls > cases
> plot(roc_rf, print.auc = TRUE, legacy.axes = TRUE,
+      main = "ROC Curve for Random Forest Model",
+      xlab = "False Positive Rate", ylab = "True Positive Rate",
+      print.thres = c(0.1, 0.5, 0.9))
> # Calculate precision, recall and F1 score
> TN_rf <- cm_rf$table[1,1] # True negatives
> FP_rf <- cm_rf$table[1,2] # False positives
> FN_rf <- cm_rf$table[2,1] # False negatives
> TP_rf <- cm_rf$table[2,2] # True positives
> 
> precision_rf <- TP_rf / (TP_rf + FP_rf)
> recall_rf <- TP_rf / (TP_rf + FN_rf)
> F1_score_rf <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
> 
> cat("Precision: ", precision_rf, "\nRecall: ", recall_rf, "\nF1 score: ", F1_score_rf, "\n")
Precision:  0.7505882 
Recall:  0.9845679 
F1 score:  0.8518024 
> 
> #----------------------------------------------------------------------------------------------
> # Logistic Regression ====
> 
> # install.packages("MASS")
> # install.packages("caret")
> library(caret)
> library(MASS)

Attaching package: ‘MASS’

The following object is masked from ‘package:dplyr’:

    select

> library(e1071)
> library(yardstick)
> #Fit a logistic regression model:
> model <- glm(stroke ~ ., data = strokeTrain, family = binomial())
> summary(model)

Call:
glm(formula = stroke ~ ., family = binomial(), data = strokeTrain)

Coefficients: (2 not defined because of singularities)
                           Estimate Std. Error z value Pr(>|z|)    
(Intercept)              -4.759e+00  6.447e-01  -7.383 1.55e-13 ***
gender                    1.533e-01  9.368e-02   1.637 0.101720    
age                       1.633e-01  7.457e-03  21.898  < 2e-16 ***
hypertension              2.175e-01  1.177e-01   1.848 0.064661 .  
heart_disease             1.396e+00  1.449e-01   9.638  < 2e-16 ***
ever_married              5.746e-01  1.690e-01   3.400 0.000675 ***
avg_glucose_level        -1.212e+02  3.837e+01  -3.158 0.001590 ** 
bmi                       1.280e-01  1.609e-02   7.955 1.79e-15 ***
smoking_status           -3.146e-01  4.821e-02  -6.525 6.82e-11 ***
work_typechildren         1.710e+00  5.708e-01   2.995 0.002740 ** 
work_typegov_job          3.041e-01  1.632e-01   1.863 0.062416 .  
work_typenever_worked    -5.612e+00  1.970e+02  -0.028 0.977269    
work_typeprivate          1.412e+00  1.181e-01  11.951  < 2e-16 ***
work_typeself_employed           NA         NA      NA       NA    
residence_typerural       3.650e-02  9.120e-02   0.400 0.688981    
residence_typeurban              NA         NA      NA       NA    
age_status               -1.267e+00  1.700e-01  -7.452 9.23e-14 ***
avg_glucose_level_status -5.174e-01  1.246e-01  -4.151 3.31e-05 ***
bmi_status               -9.471e-01  1.176e-01  -8.053 8.07e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 7954.8  on 5741  degrees of freedom
Residual deviance: 4202.3  on 5725  degrees of freedom
AIC: 4236.3

Number of Fisher Scoring iterations: 10

> #Make Predictions on test set + evaluate
> probabilities <- predict(model, newdata = strokeTest, type = "response")
> predicted_classes <- ifelse(probabilities > 0.25, 1, 0)
> 
> cm_lr <- confusionMatrix(as.factor(as.matrix(predicted_classes)), strokeTest$stroke)
> cm_lr_metrics <- cm_lr$byClass
> #Tune the model by selecting the best combination of hyperparameters:
> control <- trainControl(method="cv", number=5, verboseIter = FALSE)
> tuned_model <- train(stroke ~ ., data = strokeTrain, 
+                      method = "glm", 
+                      family = "binomial",
+                      trControl = control,
+                      tuneLength = 10)
Warning message:
In predict.lm(object, newdata, se.fit, scale = 1, type = if (type ==  :
  prediction from rank-deficient fit; attr(*, "non-estim") has doubtful cases
> print(tuned_model)
Generalized Linear Model 

5742 samples
  18 predictor
   2 classes: '0', '1' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 4594, 4593, 4594, 4594, 4593 
Resampling results:

  Accuracy  Kappa    
  0.843611  0.6860282

> # Calculate MSE, MAE, RMSE ----
> test_lr <- as.integer(strokeTest$stroke)
> mse_lr <- mean((test_lr - probabilities)^2)
> mae_lr <- mean(abs(test_lr - probabilities))
> rmse_lr <- sqrt(mean((test_lr - probabilities)^2))
> cat("MSE_lr: ", mse_lr, "\nMAE_lr: ", mae_lr, "\nRMSE_lr: ", rmse_lr)
MSE_lr:  1.496392 
MAE_lr:  1.135984 
RMSE_lr:  1.223271> 
> # Visualize CM and ROC ----
> library(ggplot2)
> library(ROCR)
> # Confusion Matrix plot
> cmv_lr <- conf_mat(table(strokeTest$stroke, predicted_classes))
> autoplot(cmv_lr, type = 'heatmap', label = TRUE) +
+     scale_fill_gradient(low = "#F3E5AB", high = "#beaed4") +
+     labs(title = paste("Confusion Matrix\nAccuracy:", round(cm_lr$overall["Accuracy"], 3)),
+          x = "Predicted Class", y = "Actual Class") +
+     theme(plot.title = element_text(hjust = 0.5))
Scale for fill is already present.
Adding another scale for fill, which will replace the existing scale.
> # ROC curve plot
> roc_lr <- roc(strokeTest$stroke, probabilities)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
> # Plot ROC curve and calculate AUC
> plot(roc_lr, main = "ROC Curve for Tuned Logistic Regression Model\n",
+      print.auc = TRUE, col = "black")
> # Calculate precision, recall and F1 score
> TN_lr <- cm_lr$table[1,1] # True negatives
> FP_lr <- cm_lr$table[1,2] # False positives
> FN_lr <- cm_lr$table[2,1] # False negatives
> TP_lr <- cm_lr$table[2,2] # True positives
> 
> precision_lr <- TP_lr / (TP_lr + FP_lr)
> recall_lr <- TP_lr / (TP_lr + FN_lr)
> F1_score_lr <- 2 * precision_lr * recall_lr / (precision_lr + recall_lr)
> 
> cat("Precision: ", precision_lr, "\nRecall: ", recall_lr, "\nF1 score: ", F1_score_lr, "\n")
Precision:  0.7364706 
Recall:  0.7017937 
F1 score:  0.7187141 
> # ----------------------------------model comparison--------------------------------------------
> Accuracy <- c(cm_1$overall[1],cm_gbm$overall[1],
+               cm_dt$overall[1],cm_svm$overall[1],
+               cm_rf$overall[1],cm_lr$overall[1])
> 
> MSE <- c(mse_1, mse_gbm, mse_dt, mse_svm, mse_rf, mse_lr)
Error: object 'mse_gbm' not found
> mse_gbm <- mean((as.numeric(test_y) - as.numeric(pred_y))^2)
> 
> Accuracy <- c(cm_1$overall[1], cm_gbm$overall[1],
+               cm_dt$overall[1], cm_svm$overall[1],
+               cm_rf$overall[1], cm_lr$overall[1])
> 
> MSE <- c(mse_1, mse_gbm, mse_dt, mse_svm, mse_rf, mse_lr)
> 
> model_comparison <- data.frame(Model = c("KNN", "LightGBM", "Decision Tree", "SVM", "Random Forest", "Logistic Regression"),
+                                Accuracy = Accuracy, MSE = MSE)
> 
> print(model_comparison)
                Model  Accuracy       MSE
1                 KNN 0.7668687 0.2331313
2            LightGBM 0.5151515 0.2126443
3       Decision Tree 0.8125253 0.1874747
4                 SVM 0.7280808 0.2719192
5       Random Forest 0.8654545 1.3793939
6 Logistic Regression 0.7030303 1.4963924
> 
> MAE <- c(mae_1, mae_gbm, mae_dt, mae_svm, mae_rf, mae_lr)
Error: object 'mae_gbm' not found
> # Calculate MSE and MAE for LightGBM
> mse_gbm <- mean((as.numeric(test_y) - as.numeric(pred_y))^2)
> mae_gbm <- mean(abs(as.numeric(test_y) - as.numeric(pred_y)))
> 
> # Now that mse_gbm and mae_gbm are defined, you can proceed with the comparisons.
> 
> # Create model comparison data frame
> model_comparison <- data.frame(
+     Model = c("KNN", "LightGBM", "Decision Tree", "SVM", "Random Forest", "Logistic Regression"),
+     Accuracy = Accuracy, 
+     MSE = MSE,
+     MAE = MAE
+ )
Error in data.frame(Model = c("KNN", "LightGBM", "Decision Tree", "SVM",  : 
  arguments imply differing number of rows: 6, 0
> MAE <- c(mae_1, mae_gbm, mae_dt, mae_svm, mae_rf, mae_lr)
> RMSE <- c(rmse_1, rmse_gbm, rmse_dt, rmse_svm, rmse_rf, rmse_lr)
Error: object 'rmse_gbm' not found
> # Calculate RMSE for LightGBM
> rmse_gbm <- sqrt(mse_gbm)
> 
> MAE <- c(mae_1, mae_gbm, mae_dt, mae_svm, mae_rf, mae_lr)
> RMSE <- c(rmse_1, rmse_gbm, rmse_dt, rmse_svm, rmse_rf, rmse_lr)
> Precision <- c(precision_1, precision_gbm, precision_dt, precision_svm, precision_rf, 
+                precision_lr)
> Recall <- c(recall_1, recall_gbm, recall_dt, recall_svm, recall_rf, recall_lr)
> F1_score <- c(F1_score_1, F1_score_gbm, F1_score_dt, F1_score_svm, F1_score_rf, F1_score_lr)
> Model <- c('KNN 1','LightGBM',
+            'Decision Tree', 'SVM',
+            'Random Forest', 'Logistic Regression')
> model_compare <- data.frame(Model, Accuracy)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('Accuracy Comparison of Models') +
+     xlab('Models') + ylab('Overall Accuracy') + theme_minimal() + 
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> #gbm#3lr#3gbm
> model_compare <- data.frame(Model, MSE)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('MSE Comparison of Models') +
+     xlab('Models') + ylab('MSE') + theme_minimal() + 
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> model_compare <- data.frame(Model, MAE)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('MAE Comparison of Models') +
+     xlab('Models') + ylab('MAE') + theme_minimal() + 
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> 
> model_compare <- data.frame(Model, RMSE)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('RMSE Comparison of Models') +
+     xlab('Models') + ylab('RMSE') + theme_minimal() + 
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> model_compare <- data.frame(Model, Precision)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('Precision Comparison of Models') +
+     xlab('Models') + ylab('Precision') + theme_minimal() + 
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> model_compare <- data.frame(Model, Recall)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('Recall Comparison of Models') +
+     xlab('Models') + ylab('Recall') + theme_minimal() + 
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> model_compare <- data.frame(Model, F1_score)
> ggplot(aes(x=Model, y=Accuracy, fill=Model), data=model_compare) +
+     geom_bar(stat='identity') + ggtitle('F1 score Comparison of Models') +
+     xlab('Models') + ylab('F1 score') + theme_minimal() +
+     theme(legend.position = "bottom", plot.title = element_text(hjust=0.5, face = "bold"))
> 
> # Random Forest Model Enhancement ====
> featureselect <- strokeTrain %>% 
+     dplyr::select(gender,age,ever_married,stroke,heart_disease,bmi,avg_glucose_level)
> featureselect_test<- strokeTest %>% 
+     dplyr::select(gender,age,ever_married,stroke,heart_disease,bmi,avg_glucose_level)
> rf_model <- randomForest(stroke ~ ., data = featureselect)
> write.csv(featureselect, file = "strokeTrain.csv", row.names = TRUE)
> #Make predictions on test set
> rf_preds <- predict(rf_model,strokeTest)
> 
> library(pROC)
> rf_preds <- as.numeric(as.character(rf_preds))
> rf_preds <- ordered(rf_preds, levels = c("0", "1"))
> rf_auc <- auc(strokeTest$stroke, rf_preds)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
> 
> # Evaluate the accuracy of the random forest model
> # Convert rf_preds to numeric
> rf_preds <- as.numeric(as.character(rf_preds))
> # Evaluate the accuracy of the random forest model
> rf_acc <- sum(rf_preds == strokeTest$stroke) / nrow(strokeTest)
> print(rf_acc)
[1] 0.7842424
> # Calculate MSE, MAE, RMSE ----
> test_rf <- as.integer(strokeTest$stroke)
> mse_rf <- mean((test_rf - rf_preds)^2)
> mae_rf <- mean(abs(test_rf - rf_preds))
> rmse_rf <- sqrt(mean((test_rf - rf_preds)^2))
> cat("MSE_rf: ", mse_rf, "\nMAE_rf: ", mae_rf, "\nRMSE_rf: ", rmse_rf)
MSE_rf:  1.526061 
MAE_rf:  1.155152 
RMSE_rf:  1.235338> 
> # Create confusion matrix
> cm_rf <- confusionMatrix(as.factor(rf_preds), strokeTest$stroke)
> library(ggplot2)
> library(ROCR)
> # Confusion Matrix plot
> cmv_rf <- conf_mat(table(strokeTest$stroke, rf_preds))
> autoplot(cmv_rf, type = 'heatmap', label = TRUE) +
+     scale_fill_gradient(low = "#F3E5AB", high = "#F1A680") +
+     labs(title = paste("Confusion Matrix\nAccuracy:", round(cm_rf$overall["Accuracy"], 3)),
+          x = "Predicted Class", y = "Actual Class")+
+     theme(plot.title = element_text(hjust = 0.5))
Scale for fill is already present.
Adding another scale for fill, which will replace the existing scale.
> 
> # ROC curve plot
> roc_rf <- roc(strokeTest$stroke, rf_preds, levels = rev(levels(strokeTest$stroke)))
Setting direction: controls > cases
> plot(roc_rf, print.auc = TRUE, legacy.axes = TRUE,
+      main = "ROC Curve for Random Forest Model",
+      xlab = "False Positive Rate", ylab = "True Positive Rate",
+      print.thres = c(0.1, 0.5, 0.9))
> 
> # Calculate precision, recall and F1 score
> TN_rf <- cm_rf$table[1,1] # True negatives
> FP_rf <- cm_rf$table[1,2] # False positives
> FN_rf <- cm_rf$table[2,1] # False negatives
> TP_rf <- cm_rf$table[2,2] # True positives
> 
> precision_rf <- TP_rf / (TP_rf + FP_rf)
> recall_rf <- TP_rf / (TP_rf + FN_rf)
> F1_score_rf <- 2 * precision_rf * recall_rf / (precision_rf + recall_rf)
> cat("Precision: ", precision_rf, "\nRecall: ", recall_rf, "\nF1 score: ", F1_score_rf, "\n")
Precision:  0.64 
Recall:  0.9158249 
F1 score:  0.7534626 
> write.csv(strokeTrain, file = "strokeTrain.csv", row.names = FALSE)
> write.csv(strokeTest, file = "strokeTest.csv", row.names = FALSE) code here...
