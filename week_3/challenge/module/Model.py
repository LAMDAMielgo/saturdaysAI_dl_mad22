from keras.applications.mobilenet import MobileNet
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Input
from tensorflow.keras.losses      import CosineSimilarity
from tensorflow.keras.metrics     import RootMeanSquaredError


def get_transfer_layer(
    train_params
  ):
  """ Defines the TRANSFER LAYER from MobileNet()
      doc: https://keras.io/api/applications/mobilenet/

  """
  
  transfer_layer = MobileNet(
      weights='imagenet',
      include_top=False,
      input_shape=train_params.get('input_shape')
  )
  # stablishes which layers to train by index


  if train_params.get('trainable_index'):
    i = train_params.get('trainable_index');print(i)

    for layer in transfer_layer.layers[:i]: layer.trainable=False
    for layer in transfer_layer.layers[i:]: layer.trainable=True

  else:
    transfer_layer.trainable = False

  return transfer_layer  


def get_top_layer(
    train_params
  ):
  """
  Defines TOP LAYER as a Secuential 

  Args:
    dropout_param
    momentum_param
  
  Return: 
    model
  """

  top_layer = Sequential(
      [
        GlobalAveragePooling2D(name='top_avgpool'),
        Flatten(),
        BatchNormalization(momentum=train_params.get('momentum'), name='top_norm'),
        Dropout(train_params.get('dropout'), name = 'top_dropout'),
        Dense(32, activation = 'relu'),
        Dense(18, activation = 'relu'),        
        Dense(train_params.get('n_classes'), activation = 'linear', name = 'top_dense_out')
      ], name = 'top_layer'
  )
  
  return top_layer


def get_model(
    train_params: dict,
    ):
  """ Model defined through Sequential() keras API with training
  params as argument
  """
  # instanciate models
  model = Sequential()  

  model.add(Input(
      train_params.get('input_shape')
  ))  

  model.add(get_transfer_layer(train_params))  
  model.add(get_top_layer(train_params))

  optimizer = Adam(learning_rate = train_params["learning_rate"])
  loss = CosineSimilarity() 
  
  model.compile(
      loss = loss,
      optimizer = optimizer,
      metrics = [RootMeanSquaredError()]
  )

  model.summary()
  return model


# ---------------------------------------------------------------------------------
# end