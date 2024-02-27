# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:20:25 2024

@author: User
"""
import tensorflow as tf

def cov_RLfBM(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability. Eq. 3.5 in Hybrid scheme for Brownian semi-stationary processes paper by Bannedsen, Lunded and Pakkenen.

        Args:
          a - float, H-1/2, where H is hurst parameter.
          n - float, the number of steps per year.
    
        Returns:
          float32 tensor
    """
    val = 1./((a+1.) * tf.pow(n, (a+1.)) )

    return tf.squeeze([[1./n, val], [val, 1./((2.*a+1.) * tf.pow(n, (2.*a+1.)) )]])


def power_kernel(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.

        Args:
          x - float32.
          a - float32, H-1/2, where H is hurst parameter.
    
        Returns:
        float32,
    """
    return tf.pow(x,a)

def optimal_discretizastion(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.

        Args:
          k - float, BUT, conceptually it should be an integer which indicates the time interval for which we calculate optimal discretisation. Float is used because types in pow function have to match.
          a - float, H-1/2, where H is hurst parameter.
        
        Returns:
        float32,
    """
    return tf.pow( (tf.pow(k,a+1) - tf.pow(k-1, a+1))/(a+1), 1/a )

def dW1(a, N_MC_paths, n_steps_per_year,T):
    """
    Produces random numbers for variance process with required
    covariance structure.

        Args:
          cov_mtr - tensor, output of  cov(a, n) function.
          N_MC_paths - int, the number of Monte Carlo samples.
          n_steps_per_year - int, the number of discretisation steps per year.
          T - float, the length of time interval in years.
    
        Returns:
          float32, tensor(N_MC_paths, n_steps_per_year)
    """

    casted_n_stps = tf.cast(n_steps_per_year, dtype=tf.float32)
    n_stps = tf.cast(casted_n_stps*T, dtype=tf.int32)
    cov_mtr = cov_RLfBM(a, casted_n_stps)

    chol = tf.linalg.cholesky(cov_mtr) # Cholesky decomposition of covariance matrix

    # 1. Generate iid normal and then normals with certain covariance structure via Cholesky.
    # 2. Reshape according to the number of Monte Carlo samples and given discretisation.
    rv = tf.reshape( tf.transpose(tf.tensordot(chol, tf.random.normal((n_stps*N_MC_paths,2), mean=0.0, stddev=1.0, dtype=tf.float32),(1,1))), (N_MC_paths,n_stps,2) )

    return rv

def Y(dW, a, n_steps_per_year, T):
    """
    Constructs Volterra process from appropriately correlated 2d Brownian increments.

        Args:
          dW - tensor, output of dW1() function.
          a - float, H-1/2, where H is hurst parameter.
          n_steps_per_year - int, the number of discretisation steps per year.
          T - float, the length of time interval in years.
    
        Returns:
        float32, tensor

    """

    casted_n_stps = tf.cast(n_steps_per_year, dtype=tf.float32)
    total_stps = tf.cast(casted_n_stps*T,dtype=tf.int32)

    #paddings = tf.constant([[0, 0,], [0, 1]])
    exact_integral = dW[:,:,1] #tf.pad(dW[:,:,1], paddings, "CONSTANT") # Exact integrals, assuming that kappa = 1.

    # Construct arrays for convolution
    # kernel is a variable which contains optimal discretization point of interval [0,T]

    kernel = tf.pad(power_kernel(optimal_discretizastion(  tf.cast(tf.range(2, total_stps+1, 1), dtype=tf.float32 ), a )/casted_n_stps, a), [[1,0]], "CONSTANT")

    # Prepare tensor of the form [dW(n),dW(n-1), ... dW(1), 0,0..0], where dW(n) = W(n)-W(n-1) increment of Wiener process.
    wiener_inct =  dW[:,:,0][:,::-1]
    wiener_inct = tf.pad( wiener_inct, [[0,0],[0,total_stps-1]], "CONSTANT") # Xi, Wiener process increment

    # Prepare variables for convolution step
    # In particular, reshape tensors.
    kernel = tf.reshape(kernel, [total_stps, 1, 1], name='kernel')
    wiener_inct = tf.reshape(wiener_inct, [tf.shape(wiener_inct)[0], tf.shape(wiener_inct)[1], 1], name='wiener_inct')

    # Tensor Flow Convolution Step (different from regular convolution !!!)
    rieman_integral = tf.squeeze( tf.nn.conv1d(wiener_inct, kernel, 1, 'VALID') )[:,::-1] #Riemann sums

    # Finally, contruct and return full process
    Y = tf.sqrt(2*a + 1) * (exact_integral + rieman_integral)
    return Y

def dW2(N_MC_paths, n_steps_per_year, T):
    """
    Outputs orthogonal increments of Wiener process.

        Args:
          N_MC_paths - int, the number of Monte Carlo samples.
          n_steps_per_year - int, the number of discretisation steps per year.
          T - float, the length of time interval in years.
    
        Returns:
        float32, tensor
    """
    casted_n_stps = tf.cast(n_steps_per_year, dtype=tf.float32)
    dt = 1.0/casted_n_stps
    n_stps = tf.cast(casted_n_stps*T, dtype=tf.int32)

    return tf.random.normal((N_MC_paths,n_stps), mean=0.0, stddev=1.0, dtype=tf.float32)*tf.sqrt(dt)

def dB(dW1, dW2, rho):
    """
    Constructs correlated Wiener increments.

        Args:
          dW1 - tensor, output of dW1() function.
          dW2 - tensor, output of dW2() function.
          rho - float32, correlation between Wiener increments.
        
        Returns:
          float32, tensor

    """
    return rho * dW1 + tf.sqrt(1 - tf.pow(rho,2) )*dW2

def V( Y, a, T, var0 , vol_of_vol, n_steps_per_year):
    """
    Outputs rBergomi variance process.

        Args:
          Y - float32 tensor, output of Y(dW, a, n_steps_per_year, T) function.
          a - float32, H-1/2, where H is hurst parameter.
          T - float32, the length of time interval in years.
          var0 - float32, initial value of variance.
          vol_of_vol - float32, volatility of volatility.
          n_steps_per_year - int, the number of discretisation steps per year.
        
        Returns:
          float32, tensor
    """
    casted_n_stps = tf.cast(n_steps_per_year, dtype=tf.float32)
    t = tf.cast(tf.linspace(0., T, tf.cast(casted_n_stps*T, tf.int32)+1),tf.float32)
    Y_padded = tf.pad(Y,[[0,0],[1,0]]) # Output of Y() defined for {t1,t2, ..., tn} and t0 we manually set it to zero.

    V = var0*tf.exp(vol_of_vol*Y_padded - 0.5*tf.pow(vol_of_vol,2)*tf.pow(t, 2*a + 1) )
    return V

def S( V, dB, S0, rho, dt ):
    """
    Outputs rBergomi realisation of price/underlying process.

        Args:
          V - float32 tensor, output of V( Y, a, T, var0 , vol_of_vol, n_steps_per_year) function.
          dB - float32 tensor, output of dB(dW1, dW2, rho) function.
          S0 - float32, initial value of the underlying.
          rho - float32, correlation between Wiener increments. 
          dt - float32, discretization step of [0,T] interval.
    
        Returns:
          float32, tensor
    """

    # Construct non-anticipative Riemann increments
    increments = tf.sqrt(V[:,:-1])*dB - 0.5*V[:,:-1]*dt

    # Cumsum is a little slower than Python loop.
    integral = tf.cumsum(increments, axis = 1)
    S = S0* tf.exp(tf.pad(integral, [[0,0],[1,0]], "CONSTANT" ))
    return S

def variance_integrals(var_process, dW1, dt):
    """
    Function calculates \int_0^t sigma_t dW_t and \int_0^t sigma_t^2 dt.

        Args:
          var_process - float32 tensor, output of  V( Y, a, T, var0 , vol_of_vol, n_steps_per_year) function.
          dW1 - float32 tensor, Wiener process increment wrt to which we calculate volatility path integrals.
          dt - float32, discretization step of [0,T] interval.
        
        Returns:
          float32 tensor, float32 tensor
    """

    # \int_0^t sigma_t dW_t, where W_t is Wiener process driving variance.
    int_voldW = tf.reduce_sum( tf.sqrt(var_process[:,:-1])*dW1[:,:,0], axis = 1 )

    # \int_0^t sigma_t^2 dt, where W_t is Wiener process driving variance.
    int_vol2dt =  tf.reduce_sum( var_process[:,:-1], axis = 1 )*dt

    return int_voldW, int_vol2dt

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ],

)
def r_Bergomi_model( S0, var0, vol_of_vol, rho, h, T,  n_steps_per_year, MC_paths ):
    """
    rBergomi price process.

        Args:
          S0 - float32, initial value of the underlying.
          var0 - float32, initial value of variance.
          vol_of_vol - float32, volatility of volatility.
          rho - float32, correlation between Wiener increments.
          h - float32, Hurst parameter.
          T - float32, the length of time interval in years.
          n_steps_per_year - int, the number of discretisation steps per year.
          MC_paths - int, the number of Monte Carlo samples.
    
        Returns:
          float 32 tensor, float 32 tensor, float 32 tensor
    """
    print('Tracing')

    alpha = h-0.5
    dt = 1.0/tf.cast(n_steps_per_year,tf.float32)

    wiener_inc1 = dW1(alpha, MC_paths, n_steps_per_year, T)
    wiener_inc2 = dW2(MC_paths, n_steps_per_year, T)
    wiener_inc_cor = dB(wiener_inc1[:,:,0], wiener_inc2, rho)

    # Truncated Wiener Process
    tr_brwn_mtn = Y(wiener_inc1, alpha, n_steps_per_year, T)

    # Variance Process
    var_proc = V( tr_brwn_mtn, alpha, T, var0, vol_of_vol, n_steps_per_year)

    # Price Process
    price_proc = S( var_proc, wiener_inc_cor, S0, rho, dt )

    # Wiener integral - int_0^T sigma_s dW_s and Rieman integral - int_0^T sigma_s^2 ds
    int_voldW,int_vol2dt = variance_integrals(var_proc, wiener_inc1, dt)

    return  price_proc, var_proc, int_voldW, int_vol2dt #tf.reduce_mean(tf.reduce_mean(price_proc,axis=1)), tf.reduce_mean(tf.reduce_mean(var_proc,axis=1)) #price_proc, var_proc