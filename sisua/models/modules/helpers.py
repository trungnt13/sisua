from functools import partial

from odin.networks import Identity
from sisua.models.modules.networks import ConvNetwork, DenseNetwork


def create_encoder_decoder(input_dim,
                           hdim=128,
                           nlayers=2,
                           xdrop=0.3,
                           edrop=0,
                           zdrop=0,
                           ddrop=0,
                           batchnorm=True,
                           linear_decoder=False,
                           activation='relu',
                           pyramid=False,
                           use_conv=0,
                           kernel=5,
                           stride=2,
                           seed=8,
                           **kwargs):
  r"""
  Arguments:
    use_conv : an integer,
      0 - DenseNetwork for both encoder and decoder
      1 - ConvNetwork for encoder, DenseNetwork for decoder
      2 - ConvNetwork for both encoder and decoder
  """
  use_conv = int(use_conv)
  network = partial(ConvNetwork, kernel=kernel,
                    stride=stride) if use_conv > 0 else DenseNetwork
  encoder = network(input_dim=input_dim,
                    units=hdim,
                    nlayers=nlayers,
                    activation=activation,
                    batchnorm=batchnorm,
                    input_dropout=xdrop,
                    output_dropout=edrop,
                    seed=seed,
                    decoding=False,
                    pyramid=pyramid,
                    name='Encoder')
  edim = encoder.output_shape[-1]
  if linear_decoder:
    decoder = Identity(name='LinearDecoder')
  else:
    network = partial(ConvNetwork, kernel=kernel,
                      stride=stride) if use_conv > 1 else DenseNetwork
    decoder = network(units=hdim,
                      nlayers=nlayers,
                      activation=activation,
                      batchnorm=batchnorm,
                      input_dropout=zdrop,
                      output_dropout=ddrop,
                      seed=seed,
                      decoding=True,
                      pyramid=pyramid,
                      name='Decoder',
                      input_dim=edim if use_conv > 1 else None)
  return encoder, decoder
