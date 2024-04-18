# from pytube import YouTube

# def extract_audio_from_url(url, output_folder):
#     yt = YouTube(url)
#     audio_stream = yt.streams.filter(only_audio=True).first()

#     # 오디오 스트림 다운로드
#     audio_stream.download(output_path=output_folder, filename=f"{yt.video_id}.wav")

# # URL 리스트
# url_list = [
#     # "https://www.youtube.com/watch?v=GAy1NSzjxYY",
#     # "https://www.youtube.com/watch?v=VIDEO_ID_2",
#     "https://www.youtube.com/watch?v=ABfQuZqq8wg"
#     # 추가 URL 추가
# ]

# # 출력 폴더
# output_folder = "C:\EDGE\SMPL-to-FBX\motions"

# # 각 URL에서 오디오 추출
# for url in url_list:
#     extract_audio_from_url(url, output_folder)

import os
import subprocess

def extract_audio_from_url(url, output_folder):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # youtube-dl 명령어 생성
    command = [
        "youtube-dl",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", f"{output_folder}/%(id)s.%(ext)s",
        url
    ]

    # youtube-dl 실행
    subprocess.run(command)

# URL 리스트
url_list = [
    # "https://www.youtube.com/watch?v=VIDEO_ID_1",
   "https://www.youtube.com/watch?v=ABfQuZqq8wg"
    # 추가 URL 추가
]

# 출력 폴더
output_folder = "C:\EDGE\SMPL-to-FBX\motions"

# 각 URL에서 오디오 추출
for url in url_list:
    extract_audio_from_url(url, output_folder)