# import pickle
# import joblib

# pickle_file_path = 'C:\_data\project\pickle\\1.pkl'

# loaded_data = joblib.load(pickle_file_path)



# print(loaded_data[2].keys)

###########################################
import pickle
import joblib
# pickle 파일 경로 설정
# pickle_file_path = 'C:\_data\project\pickle\\1234.pkl'
pickle_file_path = 'C:\_data\project\pickle\\jun123.pkl'

# pickle 파일 열기
# with open(pickle_file_path, 'rb') as f:
#     # pickle 파일에서 데이터 로드
#     loaded_data = pickle.load(f)

# # 로드된 데이터 출력
# edge = loaded_data

loaded_data = joblib.load(pickle_file_path)
print(loaded_data.keys())
# print(loaded_data.keys())
# # print(loaded_data.keys)
# print(loaded_data[0].keys())
# print(loaded_data[0]['keypoints'])
# print(loaded_data[12]['pose'].shape)
# print(loaded_data[0]['trans'].shape)


