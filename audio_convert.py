from pydub import AudioSegment

# 读取音频文件，自行修改文件路径文件名
audio = AudioSegment.from_file("./speaker/speaker_mine.mp3")

# 转换为单声道并调整采样率
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)  # 如果需要将采样率调整为 16kHz

# 保存转换后的音频文件
audio.export("./speaker/speaker_mine_converted.wav", format="wav")
