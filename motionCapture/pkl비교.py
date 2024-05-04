import pickle
import joblib
import numpy as np
# pickle 파일 경로 설정
# pickle_file_path = 'C:\_data\project\pickle\\0425_test\\bo.pkl'
pickle_file_path = 'C:\practiceField\\test_o8RkbHv2_a0.pkl'

# pickle 파일 열기
# with open(pickle_file_path, 'rb') as f:
#     # pickle 파일에서 데이터 로드
#     loaded_data = pickle.load(f)

# # 로드된 데이터 출력
# edge = loaded_data

loaded_data = joblib.load(pickle_file_path)


print(loaded_data.keys())
print(loaded_data['smpl_poses'].shape)



# print(loaded_data['smpl_poses'].shape) # (2700, 72)
# print(loaded_data['smpl_trans'].shape) # (2700, 3)
# print(loaded_data['full_pose'].shape) #  (2700, 24, 3)



# result = 0

# for key in loaded_data.keys():
    
#     print(loaded_data[key]['smpl_pose'].shape[0])
#     result = result + loaded_data[key]['smpl_pose'].shape[0]

# print("최종 결과 : ",result)

