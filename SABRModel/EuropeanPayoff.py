# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:08:19 2023

@author: User
"""
import numpy as np
import tensorflow as tf

from sabr_process import *


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def european_price_mc(S0, r, maturity, sigma0, vega, rho, strk, mcmSmpl, nStps, call):
    """ Outputs the price of an option on forward for a vector of strikes.

              Args:
                S0: float, initial forward price
                r: float, interest rate
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                rho: float, correlation
                strk: tensor(float,), strike prices
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]
                call: int, if 1 calculates call price if 0 put price.

              Returns:
                 float32 tensor

            """ 
    print('Tracing')
    
    # Reshape strike to [[K1],[K2],...]
    new_strike = tf.reshape( strk, [-1,1])
    # SABR path 
    underlying, volatility = sabr_process_full(S0=S0, maturity=maturity, sigma0=sigma0, vega=vega, rho=rho, 
                                                                                         mcmSmpl=mcmSmpl, nStps=nStps)

    payoff = ( underlying[:,-1]-new_strike )
        
    return tf.cond(call>0, 
                   lambda:tf.exp(-r*maturity)*tf.math.reduce_mean(tf.maximum(payoff,0),axis=1),
                   lambda: tf.exp(-r*maturity)*tf.math.reduce_mean(tf.maximum(-payoff,0),axis=1) )
    
        
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def european_price_mc_anthtc(S0, r, maturity, sigma0, vega, rho, strk, mcmSmpl, nStps, call):
    """ Outputs the price of an option on forward for a vector of strikes using antithetic variates.

              Args:
                S0: float, initial forward price
                r: float, interest rate
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                rho: float, correlation
                strk: tensor(float,), strike prices
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]
                call: int, if 1 calculates call price if 0 put price.

              Returns:
                 float32 tensor

            """ 
    print('Tracing')
    
    # Reshape strike to [[K1],[K2],...]
    new_strike = tf.reshape( strk, [-1,1])
    
    # SABR path 
    underlying, volatility = sabr_process_full_anthtc(S0=S0, maturity=maturity, sigma0=sigma0, vega=vega, 
                                                                                    rho=rho, mcmSmpl=mcmSmpl, nStps=nStps)

    payoff = ( underlying[0][:,-1]-new_strike )
    payoff_a = ( underlying[1][:,-1]-new_strike )

    
    return tf.cond(call>0, 
                  lambda:tf.exp(-r*maturity)*tf.math.reduce_mean(( tf.maximum(payoff,0)+tf.maximum(payoff_a,0) )/2,axis=1),
                  lambda: tf.exp(-r*maturity)*tf.math.reduce_mean(( tf.maximum(-payoff,0)+tf.maximum(-payoff,0) )/2,axis=1))

@tf.function
def normal_cdf(x):
  """Computes the normal density at the supplied points.

    Args:
      x: A float32 tensor at which the density is to be computed.

    Returns:
      A float32 tensor of the normal density evaluated at the supplied points.
  """

  return (1.0 + tf.math.erf(x / tf.sqrt(2.0))) / 2.0


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=[None], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def european_price_cmc(S0, r, maturity, sigma0, vega, rho, strk, mcmSmpl, nStps, call):
    """ Outputs the price of an option on forward for a vector of strikes using Conditional Monte Carlo.

              Args:
                S0: float, initial forward price
                r: float, interest rate
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                rho: float, correlation
                strk: tensor(float,), strike prices
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]
                call: int, if 1 calculates call price if 0 put price.

              Returns:
                 float32 tensor

            """ 
    print('Tracing')
    
    # Reshape strike to [[K1],[K2],...]
    new_strike = tf.reshape( strk, [-1,1])
    
    # SABR volatility path + corresponding integrals
    vol_path, intgr_vol2_dt, intgr_vol_dWt = sabr_vol_path(maturity=maturity, sigma0=sigma0, 
                                                                         vega=vega, mcmSmpl=mcmSmpl, nStps=nStps)
    # Mixing Solution
    S_eff = S0*tf.exp( rho*intgr_vol_dWt - 0.5*tf.pow(rho,2)*intgr_vol2_dt )
    vol_eff = tf.sqrt( (1-tf.pow(rho,2))*intgr_vol2_dt )
    
    d1 = tf.math.log(S_eff/new_strike)/vol_eff + 0.5*vol_eff 
    d2 =d1-vol_eff

    return tf.cond(call>0, 
    lambda:tf.math.reduce_mean( tf.exp(-r*maturity)*(S_eff*normal_cdf(d1)-new_strike*normal_cdf(d2)), axis=1 ),
    lambda: tf.math.reduce_mean( tf.exp(-r*maturity)*(-S_eff*normal_cdf(-d1)+new_strike*normal_cdf(-d2)), axis=1 ) )


#european_price_mc(S0=100., r=0., maturity=0.5, sigma0=0.1, vega=0.8, rho=0.7, 
#                      strk=tf.convert_to_tensor([99., 100., 103., 105.], dtype=tf.float32), mcmSmpl=800000, nStps=50, call=1)

#european_price_mc_anthtc(S0=100., r=0., maturity=0.5, sigma0=0.1, vega=0.8, rho=0.7, 
#                      strk=tf.convert_to_tensor([99., 100., 103., 105.], dtype=tf.float32), mcmSmpl=800000, nStps=50, call=1)

#european_price_cmc(S0=100., r=0., maturity=0.5, sigma0=0.1, vega=0.8, rho=0.7, 
#                      strk=tf.convert_to_tensor([99., 100., 103., 105.], dtype=tf.float32), mcmSmpl=800000, nStps=50, call=1)

