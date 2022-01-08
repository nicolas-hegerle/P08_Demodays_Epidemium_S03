import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, DenseNet201
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# function to generate the image datagenerator
def create_imgen(df):

    '''
    Supposes that the img files are stored in <src/sorted_img/train/>
    That the image file name is in the <img_rename> column of the df
    That the target label is in the <os> column

    Argments:
    df: the dataframe where the info is stored

    Returs:
    train_gen: training image generator
    val_gen: validation image generator
    '''

    # declare an instance of ImageDataGenerator
    img_generator = ImageDataGenerator(brightness_range=(0.5,1),
                                                                channel_shift_range=0.2,
                                                                horizontal_flip=True,
                                                                vertical_flip=True,
                                                                rescale=1./255,
                                                                validation_split=0.2)

    # instantiate the train img generator
    train_gen = img_generator.flow_from_dataframe(
        dataframe=df, # the dataframe containing the filename and label column
        directory="src/sorted_img/train/", # the directory containing the image files
        x_col="img_rename", # the name of the column with the filenames
        y_col="os", # the name of the column with the labels
        target_size=(224, 224),
        class_mode = "raw", 
        batch_size=32, 
        shuffle = True,
        subset = "training",
        seed = 123
    )

    # instantiate the test img generator
    val_gen = img_generator.flow_from_dataframe(
        dataframe=df, # the dataframe containing the filename and label column
        directory="src/sorted_img/train/", # the directory containing the image files
        x_col="img_rename", # the name of the column with the filenames
        y_col="os", # the name of the column with the labels
        target_size=(224, 224),
        class_mode = "raw", 
        batch_size=32, 
        shuffle = True,
        subset = "validation",
        seed = 123
    )

    return train_gen, val_gen

# function to instantiate the models for transfer learning
def create_model(application, free = None, activation = 'linear'):

    '''
    Instanciate transferlearning models as base models or with layers freed for training
    Only adapted to models for which instanciation arguments accept ```input_shape=(224,224,3), include_top=False, weights = 'imagenet'```
    Made for Inceptionv3, InceptionResNetv2 and DenseNet201

    Arguments:
    model: name of the model from tensorflow.keras.applications among InceptionResNetV2, InceptionV3, DenseNet201
    free: int, percent of layers you want to free Ex: 20 for 20%, 30 for 30%...
    activation: string, one of the available activation function for Dense layers

    Returns:
    an instance of the model
    '''

    if free:
        base_model = application(input_shape=(224,224,3), include_top=False, weights = 'imagenet')
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - int(free/100 * len(base_model.layers))
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        model = Sequential([
        base_model,
        MaxPool2D(),
        Flatten(),
        Dense(1, activation = activation)
        ])

        return model

    else:

        base_model = application(input_shape=(224,224,3), include_top=False, weights = 'imagenet')
        base_model.trainable = False
        
        model = Sequential([
        base_model,
        MaxPool2D(),
        Flatten(),
        Dense(1, activation = activation)
        ])

        return model

# function to generate the callbacks
def create_callbacks(checkpoint_path, tensorboard_path):

    '''
    Function to instantiate the callback functions for model training

    Arguments:
    checkpoint_path: string, path where to save the ceckpoints
    tensorboard_path: string, path where to save the tensorboard data

    Returns:
    an instance of the checkpoint callback and the tensorboard callback
    '''

    checkpoint_callback = ModelCheckpoint(filepath = checkpoint_path, save_best_only = True, monitor = 'val_loss', mode='min', save_weights_only = False)
    tensorboard_callback = TensorBoard(log_dir=tensorboard_path)

    return checkpoint_callback, tensorboard_callback

# function to trin the model on the data
def train_model(model, train_gen, val_gen, checkpoint, tensorboard, loss = 'mae', learning_rate = 0.05, epochs = 50):

    '''
    Trains the model for a given number of epochs and returns the best model based on val_loss

    Arguments:
    model: instance of the model you want to train
    train and val gen: instances of ImageDataGenerator flow from dataframes
    checkpoint: an instance of the checkoint callback function
    tensorboard: an instance of the tensorboard callback function
    loss: string, the loss function to use
    epochs: int, the number of epochs you want to train your model for
    
    Returns:
    the best model based on val_loss obtained over the nb epochs trained
    '''

    # compile the model and instaciate the callbak methods
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = loss)

    #fit the model on the data
    model.fit(train_gen, validation_data=val_gen, epochs = epochs, callbacks=[checkpoint, tensorboard])

    # reload the best model to have the weights that minimize the loss
    model = tf.keras.models.load_model(checkpoint.filepath)
    
    return model

# function to generate predictions from the image data generator
def generate_predictions(data, model, nb_iter, train_gen, val_gen, save = False, path = "src/pred_df/", dir = 'preds', file_names = ["df_train.csv", "df_val.csv"]):
    '''
    Uses a tranined tensorflow model to generate prediction over <nb_iter> of the train and validation image generator and return two dataframes linking predicted labels
    to true labels and absolute error between the two values as integers

    Arguments:
    data: a pandas dataframe of all the 'os' labels
    model: a trained tensorflow model
    train_gen: an instance of the ImageDataGenerator generating the training batches
    val_gen: an instance of the ImageDataGenerator generating the validation batches 
    test_gen: wheter to produce predictions on a test generator for 3 way hold out
    save: bool, whether you want to save the ouptu dataframes as csv or not
    If save = True:
    path: path to the folder where you want to save the files. MUST END WITH '/'
    file_names: a list of two names under which the generated dataframes will be saved. Best to end with file extension '.csv'

    Returns:
    df_train_pred, df_val_pred
    Two dataframes with colnames = ['true', 'pred', 'abs_error', 'os']
    true: true label
    pred: integer of the prediction made by the model
    abs_error: absolute error between the predicted and the true value calculate as abs(pred - true)
    os: 'overal survival' of the patient, equal to the true label but might can be associated to 0 in 'true', 'pred' or 'abs_error' if that label wasn't seen by the model
    '''

    val_pred = {'true':[], 'pred':[]}
    train_pred = {'true':[], 'pred':[]}
    for i in range(nb_iter):
        tr_im, tr_lab = next(iter(train_gen))
        train_pred['true'] += list(tr_lab)
        train_pred['pred'] += list((model.predict(tr_im)).reshape(-1,))

        val_im, val_lab = next(iter(val_gen))
        val_pred['true'] += list(val_lab)
        val_pred['pred'] += list((model.predict(val_im)).reshape(-1,))
        if i % 10 == 0 and i != 0:
            print(f"====Finished predictions over {i} iterations====")

    print(f"\n====Finished generating predictions over  {nb_iter} iterations====")
    
    df_train_pred = pd.DataFrame(train_pred)
    df_train_pred['pred'] = df_train_pred['pred'].apply(lambda x : int(x))
    df_train_pred['abs_error'] = abs(df_train_pred['pred'] - df_train_pred['true'])
    df_train_pred = df_train_pred.merge(data.drop_duplicates(subset='os')['os'], how='outer', left_on = 'true', right_on = 'os').fillna(0)
    df_train_pred = df_train_pred.astype("int64")

    df_val_pred = pd.DataFrame(val_pred)
    df_val_pred['pred'] = df_val_pred['pred'].apply(lambda x : int(x)) 
    df_val_pred['abs_error'] = abs(df_val_pred['pred'] - df_val_pred['true'])
    df_val_pred = df_val_pred.merge(data.drop_duplicates(subset='os')['os'], how='outer', left_on = 'true', right_on = 'os').fillna(0)
    df_val_pred = df_val_pred.astype("int64")

    train_path = os.path.join(path, dir, file_names[0])
    val_path = os.path.join(path, dir, file_names[1])
    dir_path = os.path.join(path, dir)

    if save:
        if os.path.exists(dir_path):
            df_train_pred.to_csv(train_path, index = False)
            df_val_pred.to_csv(val_path, index = False)
        
        else:
            os.mkdir(dir_path)
            df_train_pred.to_csv(train_path, index = False)
            df_val_pred.to_csv(val_path, index = False)

    return df_train_pred, df_val_pred


def convnet(height, width, channels):
    model = tf.keras.Sequential([
    Conv2D(
        filters = 32,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu",
        input_shape = (height, width, channels) # the input shape (height, width, channels)
    ),
    Conv2D(
        filters = 32,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    BatchNormalization(), MaxPool2D(data_format='channels_last'),
    Conv2D(
        filters = 64,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    Conv2D(
        filters = 64,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    BatchNormalization(), MaxPool2D(),
    Conv2D(
        filters = 128,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    Conv2D(
        filters = 128,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    BatchNormalization(), MaxPool2D(),
    Conv2D(
        filters = 256,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    Conv2D(
        filters = 256,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    BatchNormalization(), MaxPool2D(),
    Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    Conv2D(
        filters = 512,
        kernel_size = (3,3),
        strides = 1,
        padding = "same",
        activation = "relu"
    ),
    BatchNormalization(), MaxPool2D(),
    GlobalAveragePooling2D(), # turns multi-dimensional images into flat objects
    Dense(512, activation="relu"),
    Dropout(rate = 0.1),
    Dense(512, activation = "relu"),
    Dropout(rate = 0.1),
    Dense(1, activation = "linear")
    ]
    )

    return model