# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:55:52 2023

@author: User
"""

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
    print('Tracing')
    
    # Reshape strike to [[K1],[K2],...]
    new_strike = tf.reshape( strk, [-1,1])
    # SABR path 
    underlying, volatility, intgr_vol2_dt, intgr_vol_dWt = sabr_process_full(S0=S0, maturity=maturity, 
                                                                         sigma0=sigma0, vega=vega, rho=rho, 
                                                                         mcmSmpl=mcmSmpl, nStps=nStps)
    payoff = ( underlying[0][:,-1]-new_strike )
        
    return tf.cond(call>0, 
                   lambda:tf.exp(-r*maturity)*tf.math.reduce_mean(tf.maximum(payoff,0),axis=1),
                   lambda: tf.exp(-r*maturity)*tf.math.reduce_mean(tf.maximum(-payoff,0),axis=1) )