import pickle
import joblib
import numpy as np




# motion pkl 확인용
pickle_file_path = 'C:\\Users\\ed\Desktop\\project\AISTdataset/aist_plusplus_final\motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl'

loaded_data = joblib.load(pickle_file_path)

print(loaded_data.keys())
# print(loaded_data['smpl_loss'].shape)
print(loaded_data['smpl_poses'].shape)   # (720, 72)
print(loaded_data['smpl_trans'].shape)   # (720, 3)
print(loaded_data['smpl_scaling'].shape) # (1,)
# print(loaded_data['smpl_scaling'])       # [93.77886]


# keypoints2d pkl 확인용
# pickle_file_path = 'C:\\Users\\ed\Desktop\\project\AISTdataset/aist_plusplus_final\keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl'

# loaded_data = joblib.load(pickle_file_path)

# print(loaded_data.keys())   #dict_keys(['keypoints2d', 'det_scores', 'timestamps'])dict_keys(['keypoints2d', 'det_scores', 'timestamps'])
# print(loaded_data['keypoints2d'].shape) #   (9, 720, 17, 3)
# print(loaded_data['det_scores'].shape)  #   (9, 720)
# print(loaded_data['timestamps'].shape)  #   (720,)


# keypoints3d 확인용
# pickle_file_path = 'C:\\Users\\ed\Desktop\\project\AISTdataset/aist_plusplus_final\keypoints3d/gBR_sBM_cAll_d04_mBR0_ch01.pkl'

# loaded_data = joblib.load(pickle_file_path)

# print(loaded_data.keys())   #dict_keys(['keypoints3d', 'keypoints3d_optim'])
# print(loaded_data['keypoints3d'].shape) #(720, 17, 3)
# print(loaded_data['keypoints3d_optim'].shape) #(720, 17, 3)