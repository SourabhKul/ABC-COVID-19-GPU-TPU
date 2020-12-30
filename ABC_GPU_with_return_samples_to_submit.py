# Parallel ABC Inference for Stochastic Epidemology Model

import numpy as np
import time as time
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.device('/GPU:0')

tolerance = 2e5
num_samples = 500000
population = 60.36e6

# data for Italy, source: https://github.com/CSSEGISandData/COVID-19

country_data_train = tf.constant([[   155,    229,    322,    453,    655,    888,   1128,   1694,   2036,
           2502,   3089,   3858,   4636,   5883,   7375,   9172,  10149,  12462,
          15113,  17660,  21157,  24747,  27980,  31506,  35713,  41035,  47021,
          53578,  59138,  63927,  69176,  74386,  80589,  86498,  92472,  97689,
         101739, 105792, 110574, 115242, 119827, 124632, 128948, 132547, 135586,
         139422, 143626, 147577, 152271],
        [     2,      1,      1,      3,     45,     46,     46,     83,    149,
            160,    276,    414,    523,    589,    622,    724,    724,   1045,
           1045,   1439,   1966,   2335,   2749,   2941,   4025,   4440,   4440,
           6072,   7024,   7024,   8326,   9362,  10361,  10950,  12384,  13030,
          14620,  15729,  16847,  18278,  19758,  20996,  21815,  22837,  24392,
          26491,  28470,  30455,  32534],
        [     3,      7,     10,     12,     17,     21,     29,     34,     52,
             79,    107,    148,    197,    233,    366,    463,    631,    827,
           1016,   1266,   1441,   1809,   2158,   2503,   2978,   3405,   4032,
           4825,   5476,   6077,   6820,   7503,   8215,   9134,  10023,  10779,
          11591,  12428,  13155,  13915,  14681,  15362,  15887,  16523,  17127,
          17669,  18279,  18849,  19468]], dtype=tf.float32)

# Main Parallel ABC Kernel

@tf.function(experimental_compile=True)
def build_graph():
    num_days = tf.cast(country_data_train.shape[1], tf.int32)
    P = tf.ones(num_samples) * population
    A_0 = tf.ones(num_samples) * country_data_train[0, 0]
    R_0 = tf.ones(num_samples) * country_data_train[1, 0]
    D_0 = tf.ones(num_samples) * country_data_train[2, 0]
    param_vector = tf.transpose(tfd.Uniform(
        tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        tf.constant([1.0, 100.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0]),
        ).sample(num_samples))

    summary = tf.zeros([num_days, 3, num_samples])

    nu = tf.constant([[-1,  1,  0, 0, 0, 0],
                      [ 0, -1,  1, 0, 0, 0],
                      [ 0,  0, -1, 1, 0, 0],
                      [ 0,  0, -1, 0, 1, 0],
                      [ 0, -1,  0, 0, 0, 1]], dtype=tf.float32)

    S_store = P - param_vector[7] * A_0 - (A_0 + R_0 + D_0)
    I_store = param_vector[7] * A_0
    A_store = A_0
    R_store = R_0
    D_store = D_0
    Ru_store = tf.zeros(num_samples)

    summary = tf.tensor_scatter_nd_add(summary, [[0,0], [0,1], [0,2]],tf.stack([A_store, R_store, D_store]))


    def body(i, s, S, I, A, R, D, Ru):
        U = A + R + D
        alpha_t = param_vector[0] + (
                    param_vector[1] / (tf.constant(1.0) + tf.pow(U, param_vector[2])))
        h_1 = (S * I / P) * alpha_t
        h_2 = I * param_vector[4]
        h_3 = A * param_vector[3]
        h_4 = A * param_vector[5]
        h_5 = I * param_vector[6] * param_vector[3]
        h = tf.stack([h_1, h_2, h_3, h_4, h_5])
        Y_store = tf.clip_by_value(tf.math.floor(tfd.Normal(loc=h,scale=tf.sqrt(h)).sample()), 0.0, P)

        m = tf.matmul(tf.transpose(nu), Y_store)

        S = tf.clip_by_value(S + m[0,:], 0.0, P)
        I = tf.clip_by_value(I + m[1,:], 0.0, P)
        A = tf.clip_by_value(A + m[2,:], 0.0, P)
        R = tf.clip_by_value(R + m[3,:], 0.0, P)
        D = tf.clip_by_value(D + m[4,:], 0.0, P)
        Ru = tf.clip_by_value(Ru + m[5,:], 0.0, P)

        s = tf.tensor_scatter_nd_add(s, [[i,0], [i,1], [i,2]], tf.stack([A, R, D]))

        return i+1, s, S, I, A, R, D, Ru


    init_idx = tf.zeros([], dtype=tf.int32) + 1
    i, summary, *_ = tf.while_loop(lambda i, *_: i < num_days, body, [init_idx, summary, S_store, I_store, A_store, R_store, D_store, Ru_store])


    t_summary = tf.transpose(summary, perm=[2,1,0])
    distances = tf.norm( tf.broadcast_to( country_data_train,tf.constant([num_samples,country_data_train.shape[0],country_data_train.shape[1]], dtype=tf.int32))- t_summary, axis=2 )
    reduced_distances = tf.reduce_sum(distances, axis=1)
    acceptance_vector = reduced_distances <= tolerance
    num_accepted_samples = tf.reduce_sum(tf.cast(acceptance_vector, dtype=tf.float32), name = "num_accepted_samples")
    min_distances, min_dist_indices = tf.math.top_k(-reduced_distances, 5)
    params_to_return = tf.gather(param_vector, min_dist_indices, axis=1)
    return num_accepted_samples, params_to_return, -min_distances


     
# Warm-up xla compilation
build_graph()

# ABC inference
print("Running...")
max_runs = 30000
samples_target = 10
samples_collected = 0
num_runs = 0
start_time = time.time()
returned_samples = []
min_distances = []
tf.profiler.experimental.server.start(6009)
for step in range(max_runs):
    with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        num_accepted_samples, returned_run_params, min_distance = build_graph()
        samples_collected += num_accepted_samples
        if num_accepted_samples:
            returned_samples.append(returned_run_params)
            min_distances.append(min_distance)
        num_runs += 1
        if samples_collected >= samples_target:
            break

# Post processing
returned_samples = tf.stack(returned_samples)
min_distances = tf.stack(min_distances)
returned_samples = (tf.transpose(returned_samples, [2,0,1]))
flattened_samples = tf.reshape(returned_samples, [-1, 8])
selected_samples = tf.boolean_mask(flattened_samples, tf.reshape(min_distances, [-1]) <= 1e5)

end_time = time.time()

print("Completed in {0:.3f} seconds\n".format(end_time - start_time))
print(f"Samples collected: {samples_collected}")
print(f"Number of runs: {num_runs}")
print("Time per run: {0:.3f} milliseconds\n".format(1e3*(end_time - start_time)/num_runs))