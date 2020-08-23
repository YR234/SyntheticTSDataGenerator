from scipy.io.arff import loadarff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def read_single_arff_ucr(path, label="target"):
    data = loadarff(path)
    df_data = pd.DataFrame(data[0])
    # Convert to categorial (something the classes aren't numbers between 0 - #num_classes)
    uniqe = df_data[label].nunique()
    encoder = LabelEncoder()
    encoder.fit(df_data[label])
    encoded_Y = encoder.transform(df_data[label])
    Y = to_categorical(encoded_Y, uniqe)

    # Removing the label for X
    del df_data[label]
    X = df_data.values
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, Y
    

def read_arff_ucr(train_arff_path, test_arff_path, label="target"):
    """Reading arff files from the UCR archive

    Parameters:
    train_arff_path : str.
        Path to the train arff file. usually named DATASET_TRAIN.arff.
        
    test_arff_path : str.
        Path to the test arff file. usually named DATASET_TEST.arff.
        
    label : str, default "target"
        The name of the column for the target (the class)

    Returns
    -------
    X_train : numpy ndarray
        containing the X values for train.
    
    X_test : numpy ndarray
        containing the X values for test.
        
    y_train : numpy ndarray
        containing the Y values for train.
    
    y_test : numpy ndarray
        containing the Y values for test.

   """
    
    X_train, y_train = read_single_arff_ucr(train_arff_path, label)
    X_test, y_test = read_single_arff_ucr(test_arff_path, label)
    return X_train, X_test, y_train, y_test