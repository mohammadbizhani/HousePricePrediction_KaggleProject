# HousePricePrediction_KaggleProject
A project from kaggle for calculating home price, using XGB, MLP, CAT, Ridge models.


First, you can found the datas in (train.csv) and (test.csv). also there is another file name (data_discription) that contain the information and a explanation about each features. We can get important information from this file, like: Some of NaN values aren't real NaN and these NaN values means the target house doesn't have this option. This can help us to get the better results.

So, after dealing the NaN values, it's time to dealing with these big amount of features. for that, i tried to select the features that they have high correlation with target (Y). Then, we have to doing the dummy and standarding the numerical data. And at the end with MLP,CAT,XGB and Ridge models, we calculate the price of the houses. 
