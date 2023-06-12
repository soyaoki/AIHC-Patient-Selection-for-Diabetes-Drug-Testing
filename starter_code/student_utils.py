import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools


####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df_temp = df.copy()
    d = dict()
    d['nan'] = np.nan
    for i in range(ndc_df.shape[0]):
        code = ndc_df.iloc[i]['NDC_Code']
        name = ndc_df.iloc[i]['Non-proprietary Name']
        d[code] = name
    arr = []
    for i in range(df_temp.shape[0]):
        code = df_temp.iloc[i]['ndc_code']
        name = d[str(code)]
        arr.append(name)
    df['generic_drug_name'] = np.array(arr)
    return df

#Question 4
def select_first_encounter(df, patient_id='patient_nbr', encounter_id='encounter_id'):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''    
    df_temp = df.copy()
    df_temp = df_temp.sort_values(by=encounter_id, ascending=False)
    first_encounter_values = df_temp.groupby(patient_id)[encounter_id].tail(1).values
    first_encounter_df = df_temp[df_temp[encounter_id].isin(first_encounter_values)] 
    return first_encounter_df

#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr', test_percentage=0.2, validation_percentage=0.2):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    df_temp = df.copy()
    df_temp = df_temp.iloc[np.random.permutation(len(df_temp))]
    unique_values = df_temp[patient_key].unique()
    total_values = len(unique_values)
    
    train_size = round(total_values * (1 - test_percentage - validation_percentage))
    validation_size = round(total_values * validation_percentage)
    test_size = round(total_values * test_percentage)
    
    train = df_temp[df_temp[patient_key].isin(unique_values[:train_size])].reset_index(drop=True)
    validation = df_temp[df_temp[patient_key].isin(unique_values[train_size:train_size+validation_size])].reset_index(drop=True)
    test = df_temp[df_temp[patient_key].isin(unique_values[train_size+validation_size:])].reset_index(drop=True)
    
    print("Total number of unique patients in train = ", len(train['patient_nbr'].unique()))
    print("Total number of unique patients in validation = ", len(validation['patient_nbr'].unique()))
    print("Total number of unique patients in test = ", len(test['patient_nbr'].unique()))
    
    print("Training partition has a shape = ", train.shape) 
    print("Validation partition has a shape = ", validation.shape) 
    print("Test partition has a shape = ", test.shape)
    
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        cat_ftr_col = tf.feature_column.categorical_column_with_vocabulary_file(key=c,
                                                                                vocabulary_file=vocab_file_path,
                                                                                num_oov_buckets=1)
        cat_ftr = tf.feature_column.indicator_column(cat_ftr_col)
        output_tf_list.append(cat_ftr)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col,
                                                          default_value=default_value,
                                                          normalizer_fn=normalizer,
                                                          dtype=tf.dtypes.float32)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    
    return :
        m: the mean of the prediction probability
        s: the standard deviation of the prediction probability
    '''
    m = m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    df['pred_binary'] = (df['pred_mean'] >= 5).astype(int)
    student_binary_prediction = df['pred_binary'].values
    return student_binary_prediction
