import numpy as np
import joblib

def set_pkl(id, name):
    path = './rsc/'
    pkl = joblib.load(path + '{}.pkl'.format(name))
    # pkl2 = joblib.load(path + 'gBR_sBM_cAll_d04_mBR0_ch01.pkl')
    print(pkl[id]['frame_ids'])
    scaling = np.float32(90)
    output = {'smpl_poses' : pkl[id]['smpl_pose'][:],
            'smpl_trans' : pkl[id]['smpl_trans'][:],
            'smpl_scaling' : scaling}
    joblib.dump(output, path + '/motions/{}_{}.pkl'.format(name, id))