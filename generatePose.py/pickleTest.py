import pickle
import joblib
# pickle 파일 경로 설정
pickle_file_path = 'C:\_data\project\pickle\\1.pkl'

# pickle 파일 열기
# with open(pickle_file_path, 'rb') as f:
#     # pickle 파일에서 데이터 로드
#     loaded_data = pickle.load(f)

# # 로드된 데이터 출력
# edge = loaded_data

loaded_data = joblib.load(pickle_file_path)


print(loaded_data[2].keys())

# print(edge.keys())
# print(edge['smpl_poses'].shape) #(900, 72)  (2325, 72)          #   (4650, 72)
# print(edge['smpl_trans'].shape) #(900, 3)  (2325, 3)          #     (4650, 3)
# print(edge['full_pose'].shape) #(900, 24, 3)   (2325, 24, 3)  #     (4650, 24, 3)
# print(edge['body_pose'].shape)    