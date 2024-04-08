from moviepy.editor import VideoFileClip

# 비디오 파일 경로
video_path = 'C:\\_data\\project\\dance\\[NMIXX] _O.O_ Dance Practice.mp4'
# 오디오를 저장할 파일 경로
audio_path = 'c:\\_data\\project\\audio\\[NMIXX] _O.O_ Dance Practice_audio.mp3'

# 비디오 파일 로드
video_clip = VideoFileClip(video_path)

# 비디오에서 오디오 추출 및 저장
audio_clip = video_clip.audio
audio_clip.write_audiofile(audio_path)

# 자원 해제
audio_clip.close()
video_clip.close()

print(f"오디오가 성공적으로 추출되어 저장되었습니다: {audio_path}")