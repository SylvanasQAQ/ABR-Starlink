# buffer-based approach

import os
import sys
import numpy as np
import load_trace
import fixed_env as env
import itertools

os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
import a3c
#NN_MODEL = './models/pretrain_linear_reward.ckpt'
NN_MODEL = './models/nn_model_ep_27000.ckpt'


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = np.array([20000, 40000, 60000, 80000, 110000, 160000])  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 157.0
M_IN_K = 1000.0
TOTAL_VIDEO_CHUNKS = 157

REBUF_PENALTY = 160.0  # 1 sec rebuffering -> 160 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent

RANDOM_SEED = 42
TEST_TRACES = "./test/"

# MPC, BBA-0
abr_policy = "Pensieve"
LOG_PATH = "./test_results/Pensieve-retrain/"


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names, all_cooked_rtt = (
        load_trace.load_trace(TEST_TRACES)
    )

    net_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        all_cooked_rtt=all_cooked_rtt,
    )

    TOTAL_VIDEO_CHUNKS = net_env.get_total_chunk_len()
    os.makedirs(LOG_PATH, exist_ok=True)

    log_path = LOG_PATH + "log_sim_" + all_file_names[net_env.trace_idx]
    log_file = open(log_path, "w")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        (
            delay,
            sleep_time,
            buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = (
            VIDEO_BIT_RATE[bit_rate] / M_IN_K
            - REBUF_PENALTY * rebuf
            - SMOOTH_PENALTY
            * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate])
            / M_IN_K
        )

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(
            str(time_stamp / M_IN_K)
            + "\t"
            + str(VIDEO_BIT_RATE[bit_rate])
            + "\t"
            + str(buffer_size)
            + "\t"
            + str(rebuf)
            + "\t"
            + str(video_chunk_size)
            + "\t"
            + str(delay)
            + "\t"
            + str(entropy_)
            + "\t"
            + str(reward)
            + "\t"
            + str(SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]))
            + "\t"
            + str(REBUF_PENALTY * rebuf)
            + "\t"
            + str(bit_rate)
            + "\n"
        )
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        delay = float(delay) - env.LINK_RTT
        if abr_policy == "MPC":
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
            state[2, -1] = rebuf
            state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K / 100.0 # kilo byte / ms
            state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        elif abr_policy == "Pensieve":
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / 100.0  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 100.0  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        else:
            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = (float(video_chunk_size) / float(delay) / M_IN_K / 100.0)  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 100.0)  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        # write your own ABR algorithm here
        # buffer-based approach (BBA-0)
        if abr_policy == 'BBA-0':
            RESEVOIR, CUSHION = 5, 10
            if buffer_size < RESEVOIR:
                bit_rate = 0
            elif buffer_size >= RESEVOIR + CUSHION:
                bit_rate = A_DIM - 1
            else:
                bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)
            bit_rate = int(bit_rate)
        elif abr_policy == 'MPC':
            bit_rate = int(mpc_policy(state, video_chunk_remain, buffer_size, net_env, bit_rate))
        elif abr_policy == 'Pensieve':
            bit_rate = pensieve_policy(state)

        s_batch.append(state)
        entropy_ = 0.0
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write("\n")
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            # print(np.mean(entropy_record))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_PATH + "log_sim_" + all_file_names[net_env.trace_idx]
            log_file = open(log_path, "w")


past_bandwidth_ests = []
past_errors = []
CHUNK_COMBO_OPTIONS = []
MPC_FUTURE_CHUNK_COUNT = 5
for combo in itertools.product([0,1,2,3,4,5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)


def mpc_policy(state, video_chunk_remain, buffer_size, net_env, bit_rate):
    # ================== MPC =========================
    # send_data = 0
    curr_error = 0  # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
    if len(past_bandwidth_ests) > 0:
        curr_error = abs(past_bandwidth_ests[-1] - state[3, -1]) / float(state[3, -1])
    past_errors.append(curr_error)

    # pick bitrate according to MPC
    # first get harmonic mean of last 5 bandwidths
    past_bandwidths = state[3, -5:]
    while past_bandwidths[0] == 0.0:
        past_bandwidths = past_bandwidths[1:]
    # if ( len(state) < 5 ):
    #    past_bandwidths = state[3,-len(state):]
    # else:
    #    past_bandwidths = state[3,-5:]
    bandwidth_sum = 0
    for past_val in past_bandwidths:
        bandwidth_sum += 1 / float(past_val)
    harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidths))

    # future bandwidth prediction
    # divide by 1 + max of last 5 (or up to 5) errors
    max_error = 0
    error_pos = -5
    if len(past_errors) < 5:
        error_pos = -len(past_errors)
    max_error = float(max(past_errors[error_pos:]))
    future_bandwidth = harmonic_bandwidth / (1 + max_error)  # robustMPC here
    past_bandwidth_ests.append(harmonic_bandwidth)

    # future chunks length (try 4 if that many remaining)
    last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain)
    future_chunk_length = MPC_FUTURE_CHUNK_COUNT
    if TOTAL_VIDEO_CHUNKS - last_index < 5:
        future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index

    # all possible combinations of 5 chunk bitrates (9^5 options)
    # iterate over list and for each, compute reward and store max reward combination
    max_reward = -100000000
    best_combo = ()
    start_buffer = buffer_size
    # start = time.time()
    for full_combo in CHUNK_COMBO_OPTIONS:
        combo = full_combo[0:future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int(bit_rate)
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            index = (
                last_index + position + 1
            )  # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = (
                net_env.get_chunk_size(chunk_quality, index) / 1000000.0
            ) / future_bandwidth  # this is MB/MB/s --> seconds
            if curr_buffer < download_time:
                curr_rebuffer_time += download_time - curr_buffer
                curr_buffer = 0
            else:
                curr_buffer -= download_time
            curr_buffer += 4
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(
                VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality]
            )
            # bitrate_sum += BITRATE_REWARD[chunk_quality]
            # smoothness_diffs += abs(BITRATE_REWARD[chunk_quality] - BITRATE_REWARD[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

        reward = (
            (bitrate_sum / 1000.0)
            - (REBUF_PENALTY * curr_rebuffer_time)
            - (smoothness_diffs / 1000.0)
        )
        # reward = bitrate_sum - (8*curr_rebuffer_time) - (smoothness_diffs)
        if reward >= max_reward:
            if (best_combo != ()) and best_combo[0] < combo[0]:
                best_combo = combo
            else:
                best_combo = combo
            max_reward = reward
            # send data to html side (first chunk of best combo)
            send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
            if best_combo != ():  # some combo was good
                send_data = best_combo[0]
    bit_rate = send_data

    return bit_rate

actor = None
critic = None
RAND_RANGE = 1000
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

def pensieve_policy(state):
    action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
    action_cumsum = np.cumsum(action_prob)
    bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
    return bit_rate


if __name__ == "__main__":
    with tf.compat.v1.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                state_dim=[S_INFO, S_LEN],
                                learning_rate=CRITIC_LR_RATE)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model {} restored.", nn_model)

        main()