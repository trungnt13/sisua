from odin.networks import Identity
from sisua.models.modules.networks import DenseNetwork


def create_encoder_decoder(hdim=128,
                           nlayers=2,
                           xdrop=0.3,
                           edrop=0,
                           zdrop=0,
                           ddrop=0,
                           batchnorm=True,
                           linear_decoder=False,
                           activation='relu',
                           seed=8,
                           **kwargs):
  encoder = DenseNetwork(n_units=hdim,
                         nlayers=nlayers,
                         activation=activation,
                         batchnorm=batchnorm,
                         input_dropout=xdrop,
                         output_dropout=edrop,
                         seed=seed,
                         name='Encoder')
  if linear_decoder:
    decoder = Identity(name='LinearDecoder')
  else:
    decoder = DenseNetwork(n_units=hdim,
                           nlayers=nlayers,
                           activation=activation,
                           batchnorm=batchnorm,
                           input_dropout=zdrop,
                           output_dropout=ddrop,
                           seed=seed,
                           name='Decoder')
  return encoder, decoder
