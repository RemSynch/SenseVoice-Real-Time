import glob
import os

import pyaudio
import wave
import numpy as np
import time
from modelscope.pipelines import pipeline
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 音频参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_TIME = 60  # 最大录音时间（秒）

# VAD 参数
THRESHOLD = 1000
SILENCE_LIMIT = 2

# 声纹识别参数
SIMILARITY_THRESHOLD = 0.1  # 相似度阈值，可以根据需要调整

accumulated_audio = np.array([])  # 用于存储累积的音频数据
chunk_size = 200  # ms
sample_rate = 16000
chunk_stride = int(chunk_size * sample_rate / 1000)


# 初始化声纹识别模型
sv_pipeline = pipeline(
    task='speaker-verification',
    model='models/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

# 初始化 SenseVoiceSmall 模型
model_dir = "models/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cuda:0")
m.eval()


def is_silent(data_chunk):
    return max(data_chunk) < THRESHOLD


def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("开始监听...")
    audio_buffer = []
    silence_start = None
    is_recording = False

    while True:
        data = stream.read(CHUNK)
        audio_buffer.append(data)

        if len(audio_buffer) > RATE / CHUNK * MAX_TIME:
            audio_buffer.pop(0)

        if not is_recording:
            if not is_silent(np.frombuffer(data, dtype=np.int16)):
                print("检测到声音，开始录音...")
                is_recording = True
                silence_start = None
        else:
            if is_silent(np.frombuffer(data, dtype=np.int16)):
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_LIMIT:
                    print("检测到静音，停止录音")
                    break
            else:
                silence_start = None

    stream.stop_stream()
    stream.close()
    p.terminate()

    return b''.join(audio_buffer)

def save_audio(data, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def verify_voice(reference_audio, input_audio):
    result = sv_pipeline([reference_audio, input_audio])
    print(result)
    return result['score']  # 返回相似度得分


def transcribe_audio(audio_file):
    res = m.inference(
        data_in=audio_file,
        language="auto",
        use_itn=False,
        ban_emo_unk=False,
        **kwargs,
    )

    text = rich_transcription_postprocess(res[0][0]["text"])
    return text


# 手动热词
def replace_diy_hotword(sentence):
    # 定义一个Map，用于存储高频错词和对应的替换词
    error_hotkey_map = {
        '你冇': '蕾姆',
        '我冇': '蕾姆',
        '雷母': '蕾姆',
        '雷姆': '蕾姆',
        '蕾母': '蕾姆',
        '雷冇': '蕾姆',
        '蕾冇': '蕾姆',
        '人母': '蕾姆',
        '你悟': '蕾姆',
        '你姆': '蕾姆',
        '人冇': '蕾姆',
        '人姆': '蕾姆',
        '李慕': '蕾姆',
        # 添加更多的错词和替换词
    }
    for wrong_word, correct_word in error_hotkey_map.items():
        # 每次替换都更新 sentence
        sentence = sentence.replace(wrong_word, correct_word)
    # print("corrected_text:{}".format(sentence))
    return sentence


def main():
    # 创建保存目录（如果目录不存在）
    save_directory = "audio_logs"
    os.makedirs(save_directory, exist_ok=True)
    # 加载声纹锁示例音频
    reference_audio = "speakers/speaker_mine_converted.wav"

    max_files = 10  # 最多保留的文件数量

    while True:
        audio_data = record_audio()
        # 使用 os.path.join 来拼接保存路径
        output_filename = os.path.join(save_directory, f"recorded_audio_{int(time.time())}.wav")
        save_audio(audio_data, output_filename)

        # 检查并删除最旧的文件（如果文件数量超过 max_files）
        existing_files = glob.glob(os.path.join(save_directory, "*.wav"))
        if len(existing_files) > max_files:
            # 按文件的修改时间排序
            existing_files.sort(key=os.path.getmtime)
            # 删除最旧的文件
            os.remove(existing_files[0])

        print("正在进行声纹验证...")
        similarity = verify_voice(reference_audio, output_filename)

        if similarity >= SIMILARITY_THRESHOLD:
            print(f"声纹验证通过 (相似度: {similarity:.2f})")
            print("正在进行语音识别...")
            transcribed_text = replace_diy_hotword(transcribe_audio(output_filename))
            print("识别结果:")
            print(transcribed_text)

            log_dir = 'speak_log'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, 'log.txt'), 'a', encoding='utf-8') as f:
                if transcribed_text is not None or transcribed_text != "":
                    f.write(transcribed_text)
                    f.write("\n")
                    f.flush()  # 刷新缓冲区，确保数据写入磁盘
        else:
            print(f"声纹验证失败 (相似度: {similarity:.2f})")

        print("\n准备进行下一次录音，按 Ctrl+C 退出程序")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("程序已退出")
