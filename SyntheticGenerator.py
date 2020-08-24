import FeatureSet as fs
import sequences
import numpy as np
from compress_pickle import load, dump
import os



def get_kwargs(seq_type, num_samples, seq):
    others = ["StraightLine", "HighPeak", "Up"
            , "Down", "UpAndDown", "UpAndDownAndNormal"
            , "SmallUpHighDownAndNormal", "SmallDownHighUpAndNormal", "SinWave"]
    
    one_cycles = ["RandomWaveVersion", "ECG200"]
    
    seq_type = "other" if seq_type in others else seq_type
    seq_type = "ECG200" if seq_type in one_cycles else seq_type
    kwargs = {}
    
    kwargs["CrazyRandom"] = {
                        "num_samples": num_samples,
                        "seq_length": seq,
                        "num_cycles": 1,
                        "std": 10,
                        "starting_point": [-3, 3],
                        "y_max_value": [-3, 3]}
    
    
    kwargs["ECG200"] = {
                        "num_samples": num_samples,
                        "seq_length": seq,
                        "num_cycles": 1,
                        "std": 2,
                        "starting_point": [-3, 3],
                        "y_max_value": [-3, 3]}
    
    kwargs["Traffic"] = {
                        "num_samples": num_samples,
                        "seq_length": seq,
                        "num_cycles": 1,
                        "std": 1,
                        "starting_point": [-1, 1],
                        "y_max_value": [-2, 2]}
    
    kwargs["other"] = {
                        "num_samples": num_samples,
                        "seq_length": seq,
                        "num_cycles": 5,
                        "std": 0.5,
                        "starting_point": [-3, 3],
                        "y_max_value": [-3, 3]}
    
   
    return kwargs[seq_type]


def generate_synthetic_data(data_path, seq_lengths, samples_per_class, num_files, sequences_type = "default", features_set = "default", kwargs = "default"):
    """generating synthetic time series data with our simple, uniqe method.

    Parameters:
    data_path : str.
        Path to the where the generated data will be saved.
        
    seq_lengths : list.
        List of all time series sequences length to be generated. e.g [50, 100] will generate data with 50 and 100 sequence length.
        
    samples_per_class : int
        How many instances will be generated to each class.
        
    num_files : int
        How many files to be created. e.g if we are using 10 sequences, 10 samples per class, 2 files, we will have totally 2 * (10 * 10) samples of synthetic data.
        Why not create all samples in same file? because of memory issues. we split so that each file will contain ~500K samples.
        
    sequences_type : list or str, default = "default" 
        Which kind of sequnces type to be created. e.g ["Up", "HighPeak", "SinWave"].
        if default - take all possible sequences. this is the recommended option.
        
    features_set : list or str, default = "default" 
        Same to sequences_type but regarding which features to be used as target for training (the "y"). e.g ["Max", "Min", "Peaks"]
        if default - take all possible features. this is the recommended option.

    Returns
    -------
    None, cause all files will be written to disk.

   """
    path = data_path
    num_samples = samples_per_class
    num_times = num_files
    all_sequences_type = sequences.get_all_sequences() if sequences_type == "default" else sequences_type
    feature_set = fs.get_all_features() if features_set == "default" else features_set
    for i in range(num_times):
        for seq in seq_lengths:
            if not os.path.exists(path+str(seq)):
                os.makedirs(path+str(seq))
            if not os.path.exists(path+"{}/{}".format(seq, num_samples)):
                os.makedirs(path+"{}/{}".format(seq, num_samples))
            y = []
            for sequence_type in all_sequences_type:
                kwargs = get_kwargs(sequence_type, num_samples, seq)
                y_samples = eval("sequences.{}(**kwargs).generate_data()".format(sequence_type))
                y.append(y_samples)

            y = np.array([np.array(yi) for yi in y])
            y = y.reshape(len(y) * y[0].shape[0], y[0].shape[1])
            file_name = path + "{}/{}_part{}_x_test.gz".format(seq,seq,i)  
            dump(y, file_name)
            
            y = fs.create_regression_tastks_no_multi(y, feature_set)
            file_name = path + "{}/{}_part{}_y_test.gz".format(seq,seq,i)
            dump(y, file_name)
            y = None

