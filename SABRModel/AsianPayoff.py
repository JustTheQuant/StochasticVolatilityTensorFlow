# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:39:50 2023

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
def asian_price_mc(S0, r, maturity, sigma0, vega, rho, strk, mcmSmpl, nStps, call):
    """ Outputs the price of an Asian option with forward as an underlying.

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
    payoff = ( tf.math.reduce_mean(underlying,axis=1)-new_strike )
        
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
def asian_price_mc_anthtc(S0, r, maturity, sigma0, vega, rho, strk, mcmSmpl, nStps, call):
    """ Outputs the price of an Asian option with forward as an underlying 
        using Antithetic variates.

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
    underlying, volatility = sabr_process_full_anthtc(S0=S0, maturity=maturity, sigma0=sigma0, vega=vega, rho=rho, 
                                                                                         mcmSmpl=mcmSmpl, nStps=nStps)
    payoff = ( tf.math.reduce_mean(underlying[0],axis=1)-new_strike )
    payoff_a = ( tf.math.reduce_mean(underlying[1],axis=1)-new_strike )
        
    return tf.cond(call>0, 
                   lambda:tf.exp(-r*maturity)*tf.math.reduce_mean((tf.maximum(payoff,0)+tf.maximum(payoff_a,0))/2,axis=1),
                   lambda:tf.exp(-r*maturity)*tf.math.reduce_mean((tf.maximum(-payoff,0)+tf.maximum(-payoff_a,0))/2,axis=1))

#asian_price_mc(S0=100., r=0., maturity=0.5, sigma0=0.1, vega=0.8, rho=0.7, 
#                      strk=tf.convert_to_tensor([99., 100., 103., 105.], dtype=tf.float32), mcmSmpl=800000, nStps=50, call=1)

#asian_price_mc_anthtc(S0=100., r=0., maturity=0.5, sigma0=0.1, vega=0.8, rho=0.7, 
#                      strk=tf.convert_to_tensor([99., 100., 103., 105.], dtype=tf.float32), mcmSmpl=800000, nStps=50, call=1)
