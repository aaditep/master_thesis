import tensorflow as tf
import numpy as np
import h5py
import time




def map_func(example_proto):
    """
    map function to decode a single file
    :param example_proto: buffer extracted from the tfrecord file
    :return: decoded data
    """
    features = {'train/kappa_map': tf.io.FixedLenFeature([], tf.string),
                'train/ia_map': tf.io.FixedLenFeature([], tf.string),
                'train/label': tf.io.FixedLenFeature([], tf.string)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    kappa = tf.io.decode_raw(parsed_features['train/kappa_map'], tf.float32)
    IA = tf.io.decode_raw(parsed_features['train/ia_map'], tf.float32)
    labels = tf.io.decode_raw(parsed_features['train/label'], tf.float64)
    # because of faulty tfrecord_formatter implementation
    labels = tf.cast(labels, dtype=tf.float32)
    return tf.reshape(kappa, shape=(4, 128, 128)), tf.reshape(IA, shape=(4, 128,128)), labels


def preprocess(kappa,IA):
    """
    preprocess the data
    :param kappa: kappa map
    :param IA: IA map
    :return: preprocessed data
    """
    # reshape the data from (4, 128, 128) to (128, 128, 4)
    kappa = tf.reshape(kappa, (128, 128, 4))
    # substract the mean from each channel
    kappa_mean = tf.math.reduce_mean(kappa, axis=(0,1))
    kappa = kappa - kappa_mean

    # do the same with IA
    IA = tf.reshape(IA, (128, 128, 4))
    IA_mean = tf.math.reduce_mean(IA, axis=(0,1))
    IA = IA - IA_mean

    return kappa, IA

## Define the input directory and file pattern
#input_directory = '/path/to/tf_records_directory'
#file_pattern = 'train.tf_records'
## Get the list of input files
#input_files = tf.io.gfile.glob(f'{input_directory}/{file_pattern}')
## Create a list to store the preprocessed data
#preprocessed_data = []
# Iterate over the input files

def convert_files(file):
    """
    Process only one file 
    
    returns arrays of kappas IAs and list of labels = labelss
    """
    
    # Create a dataset from the file
    dataset = tf.data.TFRecordDataset(path_to_files)
    
    # Apply the map_func to decode the data
    decoded_data = dataset.map(map_func)
    
    #get length by thisd way because the tf_records does not have cardinality
    length = 0           
    for kappa, IA, labels in decoded_data:
        length += 1
        
    #create numpy arrays for storage
    kappas = np.empty((length,128,128,4))
    IAs = np.empty((length,128,128,4))
    labels_list = np.empty((length,2))
    
    i = 0
    
    #preprocsess the data
    for kappa, IA, labels in decoded_data:
        #substract the mean and turn to numpy
        kappa, IA = preprocess(kappa, IA)
        
        #store values
        kappas[i] = kappa
        IAs[i] = IA
        labels_list[i] = labels.numpy()
        i += 1
    
    #Save the kappas, IAs and labels to h5 files

    
    return 0

def process_single_file(file):
    # Create a dataset from the file
    dataset = tf.data.TFRecordDataset(file)
    
    # Apply the map_func to decode the data
    decoded_data = dataset.map(map_func)
    
    # Preprocess the data
    preprocessed_data = [(preprocess(kappa, IA), labels) for kappa, IA, labels in decoded_data]
    
    # Create the output directory if it doesn't exist
    output_directory = '/cluster/work/refregier/atepper/kids_450_h5'
    os.makedirs(output_directory, exist_ok=True)
    
    # Save the preprocessed data to h5 files
    for i, (data, labels) in enumerate(preprocessed_data):
        output_file = f'{output_directory}/example_{i}.h5'
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('kappa', data=data[0])
            f.create_dataset('IA', data=data[1])
            f.create_dataset('labels', data=labels)

# Specify the file to process
file_to_process = "/cluster/work/refregier/athomsen/KiDS450/tfrecords/cosmo_0.161_1.119_train.tfrecords"
# Time the processing of the file
start_time = time.time()
process_single_file(file_to_process)
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time} seconds")