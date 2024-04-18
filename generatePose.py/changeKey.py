import pickle

# # pickle 파일 경로 설정
# pickle_file_path = 'C:\_data\project\pickle\\edge.pkl'

# # pickle 파일 열기
# with open(pickle_file_path, 'rb') as f:
#     # pickle 파일에서 데이터 로드
#     loaded_data = pickle.load(f)

# # 'smpl_poses' 키를 'body_pose'로 변경
# if 'smpl_poses' in loaded_data:
#     loaded_data['body_pose'] = loaded_data.pop('smpl_poses')

# # 새로운 피클 파일 경로 설정
# new_pickle_file_path = 'C:\_data\project\pickle\\new_edge.pkl'

# # 변경된 데이터를 새로운 피클 파일에 저장
# with open(new_pickle_file_path, 'wb') as f:
#     pickle.dump(loaded_data, f)

# print("피클 파일이 성공적으로 저장되었습니다.")
# pickle 파일 경로 설정
pickle_file_path = 'C:\_data\project\pickle\\new_edge.pkl'

# pickle 파일 열기
with open(pickle_file_path, 'rb') as f:
    # pickle 파일에서 데이터 로드
    loaded_data = pickle.load(f)

# 새로운 딕셔너리 생성
new_data = {}

# 'loaded_data'의 각 키에 대해 처리
for key, value in loaded_data.items():
    # 각 키의 첫 번째 행 추출
    first_row = value[0:1, :]
    # 새로운 딕셔너리에 키와 첫 번째 행 추가
    new_data[key] = first_row

# 새로운 피클 파일 경로 설정
new_pickle_file_path = 'C:\_data\project\pickle\\first_rows.pkl'

# 새로운 딕셔너리를 새로운 피클 파일에 저장
with open(new_pickle_file_path, 'wb') as f:
    pickle.dump(new_data, f)

print("첫 번째 행들이 성공적으로 저장되었습니다.")