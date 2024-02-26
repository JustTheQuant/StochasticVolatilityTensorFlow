# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:58:00 2024

@author: User
"""
import numpy as np
import tensorflow as tf
from py_vollib.black_scholes.implied_volatility import implied_volatility

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ],

)
def BS_vega(log_S0, log_strk, r, maturity, sigma):
  """ Partial derivative of Black-Scholes wrt to volatility as a function of log-underlying and log-strike.

              Args:
                log_S0: float, log of the underlying price
                log_strk: tensor(float,), log of strike prices
                r: float, interest rate
                maturity: float, maturity of the option
                sigma: float, initial value of volatility

              Returns:
                 float32 tensor
  """
  print('Tracing')

  pi = 3.141592653589793

  new_strike = tf.reshape( log_strk, [-1,1])

  term1 = tf.math.sqrt(maturity/(2*pi))

  term2 = tf.exp(log_S0 + tf.pow((2*log_strk-2*log_S0-maturity*(2*r+tf.pow(sigma,2))),2)/(-8*maturity*tf.pow(sigma,2)) )

  res = term2*term1

  return res


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def BS_strike(log_S0, log_strk, r, sigma, maturity, call):
  """ Partial derivative of Black-Scholes wrt to strike price as a function of log-underlying and log-strike.

              Args:
                log_S0: float, log of the underlying price
                log_strk: tensor(float,), log of strike prices
                r: float, interest rate
                sigma: float, initial value of volatility
                maturity: float, maturity of the option
                call: int, 1 if call option and 0 if put option.

              Returns:
                 float32 tensor
  """
  print('Tracing')

  return tf.cond(call>0, lambda: -0.5*tf.exp(log_strk-r*maturity)*tf.math.erfc( (2*log_strk-2*log_S0-maturity*(2*r-tf.pow(sigma,2)))/(2*tf.math.sqrt(2*maturity)*sigma) ), 
                 lambda: 0.5*tf.exp(log_strk-r*maturity)*tf.math.erfc( (-2*log_strk+2*log_S0+maturity*(2*r-tf.pow(sigma,2)))/(2*tf.math.sqrt(2*maturity)*sigma) ) )

def iv_retr(S0, opt_prices, strk, r, maturity, flag='c'):
    """ Calculates Black-Scholes Implied Volatility.

    Args:
      S0 - current price of a stock
      opt_prices -array of option prices corresponding to given strikes and maturity/ies
      strk - list of strikes ([K1,K2,....Kn])
      maturity - list of values or scalar value
      flag - str, either 'c' for call option or 'p' for put option.
  
    Returns:
        float32 np.array  
    """

    res = []

    for i in range(len(strk)): # Loop over strikes, can be optimised. But for now this part is not that crucial.

        res.append( implied_volatility(price=opt_prices[i], S=S0, K=strk[i], t=maturity, r=r, flag=flag) )

    return np.array(res)