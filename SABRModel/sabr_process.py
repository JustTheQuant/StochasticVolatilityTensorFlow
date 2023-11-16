# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 22:31:36 2023

@author: User
"""
import numpy as np
import tensorflow as tf

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def sabr_process_full(S0, maturity, sigma0, vega, rho, mcmSmpl, nStps): # Graph Execution
    """ Outputs underlying and volatility paths under the SABR model.

              Args:
                S0: float, initial forward price
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                rho: float, correlation
                strk: tensor(float,), strike prices
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]

              Returns:
                 float32 tensor, float32 tensor

            """
    print('Tracing')

    dt = maturity/tf.cast(nStps , dtype=tf.float32 ) # time increment
    t =  tf.linspace(0.0, maturity, nStps+1) #time interval split

    # Simulating SABR volatility Wiener process
    paddings = tf.constant([[0, 0,], [1, 0]])
    # wiener process icrements with W0=0
    vol_wnr_inc = tf.sqrt(dt)*tf.random.normal((mcmSmpl, nStps), mean=0.0, stddev=1.0, dtype=tf.float32) 
    vol_wnr_inc = tf.pad(vol_wnr_inc, paddings, "CONSTANT")
    # Wiener process - summation is done rowise
    vol_wnr_proc = tf.cumsum(vol_wnr_inc, axis=1) 
    # Simulating SABR stock Wiener process
    #Bt = rho*Wt+sqrt(1-rho**2)*Zt
    und_wnr_inc = rho*vol_wnr_inc[:,1:] +  tf.sqrt(1-tf.pow(rho,2))*tf.sqrt(dt)*tf.random.normal((mcmSmpl, nStps), mean=0.0, stddev=1.0, dtype=tf.float32) # wiener process icrements with W0=0
    und_wnr_inc = tf.pad(und_wnr_inc, paddings, "CONSTANT")

    # REGULAR PATHS
    #Simulating SABR volatility path
    hlp = sigma0*tf.exp( -0.5*tf.pow(vega,2)*t )

    vol_path = hlp*tf.exp( vega*vol_wnr_proc )
    #Int [0,T] sigma_t**2 dt
    intgr_vol2_dt = dt*(tf.cumsum(tf.pow(vol_path,2),axis=1)[:,:nStps]) 
    #Int [0,T] sigma_t dBt, where Bt = rho*Wt+sqrt(1-rho**2)*Zt
    intgr_vol_dBt = tf.cumsum((vol_path[:,:nStps])*(und_wnr_inc[:,1:]),axis=1)

    # Simulating SABR stock path
    und_path = S0*( tf.exp( tf.pad( -0.5*intgr_vol2_dt + intgr_vol_dBt, paddings, "CONSTANT")  ) )

    return  und_path, vol_path

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def sabr_process_full_anthtc(S0, maturity, sigma0, vega, rho, mcmSmpl, nStps): # Graph Execution
    """ Outputs underlying and volatility paths and antithetic counterparts under the SABR model.

              Args:
                S0: float, initial forward price
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                rho: float, correlation
                strk: tensor(float,), strike prices
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]

              Returns:
                 [float32 tensor,float32 tensor], [float32 tensor, float32 tensor]

            """
    print('Tracing')

    dt = maturity/tf.cast(nStps , dtype=tf.float32 ) # time increment
    t =  tf.linspace(0.0, maturity, nStps+1) #time interval split

    # Simulating SABR volatility Wiener process
    paddings = tf.constant([[0, 0,], [1, 0]])
    # wiener process icrements with W0=0
    vol_wnr_inc = tf.sqrt(dt)*tf.random.normal((mcmSmpl, nStps), mean=0.0, stddev=1.0, dtype=tf.float32) 
    vol_wnr_inc = tf.pad(vol_wnr_inc, paddings, "CONSTANT")
    # Wiener process - summation is done rowise
    vol_wnr_proc = tf.cumsum(vol_wnr_inc, axis=1) 
    # Simulating SABR stock Wiener process
    #Bt = rho*Wt+sqrt(1-rho**2)*Zt
    und_wnr_inc = rho*vol_wnr_inc[:,1:] +  tf.sqrt(1-tf.pow(rho,2))*tf.sqrt(dt)*tf.random.normal((mcmSmpl, nStps), mean=0.0, stddev=1.0, dtype=tf.float32) # wiener process icrements with W0=0
    und_wnr_inc = tf.pad(und_wnr_inc, paddings, "CONSTANT")

    # REGULAR PATHS
    #Simulating SABR volatility path
    hlp = sigma0*tf.exp( -0.5*tf.pow(vega,2)*t )

    vol_path = hlp*tf.exp( vega*vol_wnr_proc )
    #Int [0,T] sigma_t**2 dt
    intgr_vol2_dt = dt*(tf.cumsum(tf.pow(vol_path,2),axis=1)[:,:nStps]) 
    #Int [0,T] sigma_t dBt, where Bt = rho*Wt+sqrt(1-rho**2)*Zt
    intgr_vol_dBt = tf.cumsum((vol_path[:,:nStps])*(und_wnr_inc[:,1:]),axis=1) 

    # Simulating SABR stock path
    und_path = S0*( tf.exp( tf.pad( -0.5*intgr_vol2_dt + intgr_vol_dBt, paddings, "CONSTANT")  ) )

    # ANTITHETIC PATHS
    #Simulating SABR volatility path
    vol_path_a = hlp*tf.exp(vega*(-vol_wnr_proc))
    #Int [0,T] sigma_t**2 dt
    intgr_vol2_dt_a = dt*(tf.cumsum(tf.pow(vol_path_a,2),axis=1)[:,:nStps]) 
    #Int [0,T] sigma_t dBt, where Bt = rho*Wt+sqrt(1-rho**2)*Zt
    intgr_vol_dBt_a = tf.cumsum((vol_path_a[:,:nStps])*(-und_wnr_inc[:,1:]),axis=1) 

    # Simulating SABR stock path
    und_path_a = S0*( tf.exp( tf.pad( -0.5*intgr_vol2_dt_a + intgr_vol_dBt_a, paddings, "CONSTANT")  ) )

    return  [und_path, und_path_a], [vol_path, vol_path_a]



@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def sabr_vol_path(maturity, sigma0, vega, mcmSmpl, nStps): # Graph Execution
    """ Outputs volatility path, Int [0,T] sigma_t**2 dt and Int [0,T] sigma_t dWt 
        (where sigma_t is measurable wrt to Wt).

              Args:
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                rho: float, correlation
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]

              Returns:
                float32 tensor, float32 tensor, float32 tensor

            """
    print('Tracing')

    dt = maturity/tf.cast(nStps , dtype=tf.float32 ) # time increment
    t =  tf.linspace(0.0, maturity, nStps+1) #time interval split

    # Simulating SABR volatility Wiener process
    paddings = tf.constant([[0, 0,], [1, 0]])
    # wiener process icrements
    vol_wnr_inc = tf.sqrt(dt)*tf.random.normal((mcmSmpl, nStps), mean=0.0, stddev=1.0, dtype=tf.float32) 
    vol_wnr_inc = tf.pad(vol_wnr_inc, paddings, "CONSTANT")
    # Wiener process - summation is done rowise
    vol_wnr_proc = tf.cumsum(vol_wnr_inc, axis=1)

    # REGULAR PATHS
    #Simulating SABR volatility path
    hlp = sigma0*tf.exp( -0.5*tf.pow(vega,2)*t )
    vol_path = hlp*tf.exp( vega*vol_wnr_proc )
    
    #Int [0,T] sigma_t**2 dt
    intgr_vol2_dt = dt*tf.reduce_sum(tf.pow(vol_path[:,:nStps],2),axis=1)
    #Int [0,T] sigma_t dWt
    intgr_vol_dWt = tf.reduce_sum( vol_path[:,:nStps]*vol_wnr_inc[:,1:], axis=1)

    return  vol_path, intgr_vol2_dt, intgr_vol_dWt
    

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def sabr_vol_path_anthtc(maturity, sigma0, vega, mcmSmpl, nStps): # Graph Execution
    """ Outputs volatility path, Int [0,T] sigma_t**2 dt and Int [0,T] sigma_t dWt and atithetic counterparts.
        (where sigma_t is measurable wrt to Wt).

              Args:
                maturity: float, maturity of the option
                sigma0: float, initial value of volatility
                vega: float, volatility of volatility
                mcSmpl: int, the number of Monte Carlo paths to generate
                n: int, the number of intervals on which we split [0,T]

              Returns:
                [float32 tensor,float32 tensor], [float32 tensor, float32 tensor], [float32 tensor, float32 tensor]

            """
    print('Tracing')

    dt = maturity/tf.cast(nStps , dtype=tf.float32 ) # time increment
    t =  tf.linspace(0.0, maturity, nStps+1) #time interval split

    # Simulating SABR volatility Wiener process
    paddings = tf.constant([[0, 0,], [1, 0]])
    # wiener process icrements with W0=0
    vol_wnr_inc = tf.sqrt(dt)*tf.random.normal((mcmSmpl, nStps), mean=0.0, stddev=1.0, dtype=tf.float32) 
    vol_wnr_inc = tf.pad(vol_wnr_inc, paddings, "CONSTANT")
    # Wiener process - summation is done rowise
    vol_wnr_proc = tf.cumsum(vol_wnr_inc, axis=1) 

    # REGULAR PATHS
    #Simulating SABR volatility path
    hlp = sigma0*tf.exp( -0.5*tf.pow(vega,2)*t )
    vol_path = hlp*tf.exp( vega*vol_wnr_proc )
    
    #Int [0,T] sigma_t**2 dt
    intgr_vol2_dt = dt*tf.reduce_sum(tf.pow(vol_path[:,:nStps],2),axis=1)
    #Int [0,T] sigma_t dWt
    intgr_vol_dWt = tf.reduce_sum( vol_path[:,:nStps]*vol_wnr_inc[:,1:], axis=1)
    
    # ANTITHETIC PATHS
    vol_path_a = hlp*tf.exp( vega*(-vol_wnr_proc) )
    
    #Int [0,T] sigma_t**2 dt
    intgr_vol2_dt_a = dt*tf.reduce_sum(tf.pow(vol_path_a[:,:nStps],2),axis=1)
    #Int [0,T] sigma_t dWt
    intgr_vol_dWt_a = tf.reduce_sum( vol_path_a[:,:nStps]*(-vol_wnr_inc[:,1:]), axis=1)

    return  [vol_path, vol_path_a], [intgr_vol2_dt, intgr_vol2_dt_a], [intgr_vol_dWt, intgr_vol_dWt_a]
    

#sabr_vol_path(maturity=0.5, sigma0=0.3, vega=0.5, mcmSmpl=10, nStps=10)
#sabr_vol_path_anthtc(maturity=0.5, sigma0=0.3, vega=0.5, mcmSmpl=10, nStps=10)
#sabr_process_full_anthtc(S0=100., maturity=0.5, sigma0=0.3, vega=0.5, rho=-0.3, mcmSmpl=10, nStps=10)
#sabr_process_full(S0=100., maturity=0.5, sigma0=0.3, vega=0.5, rho=-0.3, mcmSmpl=10, nStps=10)
