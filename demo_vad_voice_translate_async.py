import gc
import glob
import time
import numpy as np
import sounddevice as sd
import sherpa_onnx
import os
import scipy.io.wavfile as wav
import torch
from funasr import AutoModel
from concurrent.futures import ThreadPoolExecutor

g_sample_rate = 16000  # 采样率，只能16k哦
vad_model_path = './models/vad_onnx/silero_vad.onnx'  # VAD onnx模型路径
audio_save_path = './audio_logs'  # 保存临时音频文件的路径
max_files = 10  # 最多保留的临时音频数量
window_size = 512  # 语音窗口大小
min_silence_duration = 1.5  # 最小静音持续时间(s)
min_speech_duration = 0.05  # 最小语音持续时间(s)
segment_count = 0  # 用于生成音频文件的命名

# 初始化ASR、PUNC、SPK模型
# 为什么要加一个VAD模型呢在这？因为这里不加VAD的话PUNC就不生效，我也懒得再分开了，感觉也不吃性能。
# 觉得膈应的可以把AutoModel里的VAD去掉，然后把PUNC单独拆开，示例：
# model = AutoModel(model="ct-punc", model_revision="v2.0.4")
# res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
asr_model = AutoModel(model="./models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                      model_revision="v2.0.4",
                      vad_model="./models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                      vad_model_revision="v2.0.4",
                      punc_model="./models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                      punc_model_revision="v2.0.4",
                      # 想搞说话人确认的在这加进来
                      # spk_model="./models/speech_campplus_sv_zh-cn_16k-common",
                      # spk_model_revision="v2.0.2",
                      device="cuda:0")


def save_audio_to_file(audio_data, file_name):
    output_dir = os.path.dirname(file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    amplified_audio = audio_data * 2
    amplified_audio = np.clip(amplified_audio, -1.0, 1.0)
    wav.write(file_name, g_sample_rate, amplified_audio.astype(np.float32))
    print(f"Audio saved to {file_name}")


def rm_file():
    # 检查并删除最旧的文件（如果文件数量超过 max_files）
    existing_files = glob.glob(os.path.join(audio_save_path, "*.wav"))
    if len(existing_files) > max_files:
        # 按文件的修改时间排序
        existing_files.sort(key=os.path.getmtime)
        # 删除最旧的文件
        os.remove(existing_files[0])


def asr_inference_task(file_name, segment_count):
    """
    ASR 推理任务：从文件加载音频并进行识别。
    """
    try:
        asr_res = asr_model.generate(input=file_name,
                                     batch_size_s=300,
                                     hotword='蕾姆 拉姆')
        for resi in asr_res:
            print(f"[{segment_count}] 结果：{resi['text']}")
            print("--" * 150)
    finally:
        # 手动清理显存和对象
        del asr_res  # 删除结果对象
        torch.cuda.empty_cache()  # 释放显存
        gc.collect()  # 强制进行垃圾回收


def start_recording_with_vad(vad_model_path, window_size):
    global segment_count
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model_path
    vad_config.silero_vad.min_silence_duration = min_silence_duration
    vad_config.silero_vad.min_speech_duration = min_speech_duration
    vad_config.sample_rate = g_sample_rate

    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    samples_per_read = int(0.1 * g_sample_rate)

    print("--" * 150)
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
        return

    default_input_device_idx = sd.default.device[0]
    print(f'当前使用默认设备: 【{devices[default_input_device_idx]["name"]}】')
    print("开始监听...")

    buffer = np.array([])  # 初始化缓冲区
    preroll_buffer = np.array([])  # 用于存储前导音频
    preroll_duration = 0.3  # 希望保留的前导音频的持续时间（s）
    preroll_samples = int(preroll_duration * g_sample_rate)  # 希望保留的前导音频的采样点数（0.2s * 16000 = 3200个采样点）

    # 初始化线程池执行器
    with ThreadPoolExecutor(max_workers=2) as executor:
        with sd.InputStream(channels=1, dtype="float32", samplerate=g_sample_rate) as stream:
            while True:
                samples, _ = stream.read(samples_per_read)  # 每次读取 0.1 秒的音频数据
                samples = samples.reshape(-1)  # 将音频数据展平为一维数组

                # 更新前导缓冲区
                preroll_buffer = np.concatenate([preroll_buffer, samples])
                if len(preroll_buffer) > preroll_samples:
                    preroll_buffer = preroll_buffer[-preroll_samples:]

                buffer = np.concatenate([buffer, samples])  # 将新读取的音频数据添加到缓冲区

                while len(buffer) > window_size:
                    start_time = time.time()
                    vad.accept_waveform(buffer[:window_size])  # 向 VAD 模型输入窗口大小的音频数据
                    buffer = buffer[window_size:]

                    while not vad.empty():
                        end_time = time.time()
                        detection_time = end_time - start_time
                        segment = vad.front

                        if len(segment.samples) < 0.5 * g_sample_rate:
                            vad.pop()
                            continue

                        print(f"检测到语音活动, VAD检测耗时： {detection_time:.6f} s")

                        # 合并前导音频和检测到的语音段
                        full_segment = np.concatenate([preroll_buffer, np.array(segment.samples)])
                        segment_count += 1
                        file_name = f"{audio_save_path}/speech_segment_{segment_count}.wav"
                        save_audio_to_file(np.array(full_segment), file_name)

                        # 将ASR推理任务提交到线程池中进行异步处理
                        executor.submit(asr_inference_task, file_name, segment_count)

                        vad.pop()


if __name__ == "__main__":
    try:
        start_recording_with_vad(vad_model_path, window_size)
    except KeyboardInterrupt:
        print("加，马达捏~")
