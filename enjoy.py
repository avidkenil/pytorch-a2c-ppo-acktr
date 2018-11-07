import argparse
import os
import sys

import numpy as np
import torch
import pickle


from envs import VecPyTorch, make_vec_envs
from utils import get_render_func, get_vec_normalize


def save_object(object, filepath):
	'''
	This is a defensive way to write pickle.write, allowing for very large files on all platforms
	'''
	max_bytes = 2**31 - 1
	bytes_out = pickle.dumps(object, protocol=4)
	n_bytes = sys.getsizeof(bytes_out)
	with open(filepath, 'wb') as f_out:
		for idx in range(0, n_bytes, max_bytes):
			f_out.write(bytes_out[idx:idx+max_bytes])


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--non-det', action='store_true', default=False,
                    help='whether to use a non-deterministic policy')
args = parser.parse_args()

args.det = not args.non_det

env = make_vec_envs(args.env_name, args.seed + 1000, 1,
                            None, None, args.add_timestep, device='cpu',
                            allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

if render_func is not None:
    render_func('human')

obs = env.reset()

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

final_video = np.array([])
desired_frames = 10000
count = 0
try:
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        #print(type(obs.numpy()))
        save_obs = np.swapaxes(obs.numpy(),0,1)[np.newaxis,:].astype(np.int16)
        print(save_obs.shape)
        print(final_video.shape)
        final_video = np.vstack((final_video,save_obs)) if final_video.size is not 0 else save_obs
        print(final_video.shape)
        print(final_video.dtype)

        #np.save(open('obs.npy','wb'),obs)

        masks.fill_(0.0 if done else 1.0)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if render_func is not None:
            render_func('human')
        count+=1
        if(count >= desired_frames):
            save_object(final_video,'saved_videos/PongVideo.pkl')
            break

except KeyboardInterrupt:
    print('Keyboard Interupped. File will now be saved')
    save_object(final_video,'saved_videos/PongVideo.pkl')
