import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def training():
    # Load the data
    train = pd.read_csv('kitchenware-classification/data/train.csv')
    test = pd.read_csv('kitchenware-classification/data/test.csv')

    train.Id = train.Id.astype(str)
    test.Id = test.Id.astype(str)

    for i in train.Id:
        if len(str(i)) < 4:
            train.Id = train.Id.replace(i, '0'*(4-len(str(i))) + str(i)).astype(str)

    for i in test.Id:
        if len(str(i)) < 4:
            test.Id = test.Id.replace(i, '0'*(4-len(str(i))) + str(i)).astype(str)

    def append_ext(fn):
        return fn+".jpg"

    train["Id"]=train["Id"].apply(append_ext)
    test["Id"]=test["Id"].apply(append_ext)

    classes = train['label'].unique().tolist()

    logging.info("test")
    logging.info(f'Total number of classes: {train["label"].nunique()}')
    logging.info(f'Total number of images for training: {train.shape[0]}')
    logging.info(f'Total number of images for testing: {test.shape[0]}')
    logging.info(f'Total train + test images: {train.shape[0] + test.shape[0]}')
    logging.info(f'Total number of images in the images folder: {len(os.listdir("./kitchenware-classification/images"))}')


    # Training baseline model with Xception
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = train_gen.flow_from_dataframe(
        dataframe=train,
        directory='./kitchenware-classification/images/',
        x_col='Id',
        y_col='label',
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_gen.flow_from_dataframe(
        dataframe=train,
        directory='./kitchenware-classification/images/',
        x_col='Id',
        y_col='label',
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Architecture
    logging.info("Architecture")
    base_model = Xception(weights='imagenet', include_top=False)
    base_model.trainable = False
    inputs = keras.Input(shape=(299, 299, 3))
    base = base_model(inputs, training=False)
    vectors = GlobalAveragePooling2D()(base)
    outputs = Dense(6, activation='softmax')(vectors)
    model = Model(inputs, outputs)

    # Compile
    logging.info("Compile")
    learning_rate = 0.0001
    epochs = 10
    batch_size = 32
    optimizer = Adam(lr=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    # fit
    logging.info("Fit Model...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[
            ModelCheckpoint('./kitchenware-classification/models/model.h5', save_best_only=True),
            EarlyStopping(patience=3)
        ]
    )

    logging.info("Training Completed, model saved to ./models/model.h5")

def predict_single(model, image_url, web=True):
    from tensorflow import keras
    import numpy as np
    import json

    # predict from image url from the web
    import requests
    from PIL import Image
    from io import BytesIO

    if web:
        logging.info("Predicting from web image...")
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        logging.info("Predicting from local image...")
        img = Image.open(image_url)

    # preprocess image
    from tensorflow.keras.applications.xception import preprocess_input
     
    logging.info("Preprocessing image...")
    img = img.resize((299, 299))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # # load model
    # logging.info("Loading model...")
    # model = keras.models.load_model('./kitchenware-classification/models/model_kaggle.h5')

    # load class indices
    logging.info("Loading class indices...")
    with open('./kitchenware-classification/models/class_indices.json', 'r') as f:
        class_indices = json.load(f)

    # predict and decode along with probabilities
    logging.info("Predicting...")
    preds = model.predict(img)
    pred = np.argmax(preds, axis=1)
    labels = dict((v,k) for k,v in class_indices.items())
    predictions = [labels[k] for k in pred][0]
    prob = np.max(preds, axis=1)[0]
    return {'Label': predictions, 'Probability': str(prob)}


if __name__ == '__main__':
    training()
    predict_single('https://www.baladeo.com/medias/produits/1630258919/1710_1280-security-kinfe-emergency-yellow.jpg')

