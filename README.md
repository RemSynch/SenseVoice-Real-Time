# SenseVoice-Real-Time

本项目就是一个小的学习项目，实现了一下最简单的语音端点检测VAD（没用模型实现，这点确实该优化一下）、利用SenseVoice实现语音转录、利用CAM++实现说话人确认（声纹锁），实现并不复杂，简单玩玩，没什么实力，如果你找到了我这那纯属我们的缘分哈哈。



## 运行步骤

1. **下个CUDA版本的torch，版本号<=2.3就行（但是可能会下的比较慢）**

   ```
   pip install torch==2.2.1+cu118 torchaudio==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118
   ```

2. **安装requirements.txt中的依赖**

   ```
   pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
   ```

3. **前往`modelscope`下载`SenseVoiceSmall`**

   链接：https://www.modelscope.cn/models/iic/SenseVoiceSmall/files

   或者直接通过 git lfs下载

   ```
   git clone https://www.modelscope.cn/iic/SenseVoiceSmall.git
   ```

   下载完成后放入根目录下的`SenseVoiceSmall`文件夹中即可

4. **前往`modelscope`下载`iic/speech_campplus_sv_zh_en_16k-common_advanced`**
   链接：https://www.modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/files
   或者直接通过 git lfs下载

   ```
   git clone https://www.modelscope.cn/iic/speech_campplus_sv_zh_en_16k-common_advanced.git
   ```

   下载后放入根目录下的`speech_campplus_sv_zh_en_16k-common_advanced`文件夹即可

5. **自行录音，手机随便录一段，并通过脚本转换采样率**

   首先将音频放入speakers文件夹中
   项目中提供了脚本`audio_convert.py`，将音频转换为WAV格式并把采样率转换为16K，因为`speech_campplus`模型只能处理16K的音频。
   如果你的音频名字有修改，记得去`demo_record_natural_voice_lock.py`中也把文件名改一下

   ```Python
   def main():
       # 创建保存目录（如果目录不存在）
       save_directory = "audio_logs"
       os.makedirs(save_directory, exist_ok=True)
       # 加载声纹锁示例音频，如果你的音频名字修改了则这里也需要修改
       reference_audio = "speakers/speaker_mine_converted.wav"
   ```

6. 运行`demo_record_natural_voice_lock.py`

   ```
   python demo_record_natural_voice_lock.py
   ```

   

   运行示例
   ![运行截图](./pics/运行截图.png)



如果你觉得运行效果不太行，可以修改一下参数，在`demo_record_natural_voice_lock.py`中

```Python
# 音频参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
MAX_TIME = 60  # 最大录音时间（秒）

# VAD 参数
THRESHOLD = 500
SILENCE_LIMIT = 2

# 声纹识别参数
SIMILARITY_THRESHOLD = 0.1  # 相似度阈值，可以根据需要调整
```



另外我觉得SenseVoice转录错字率有点太高了，不过能识别粤语挺不错的占用显存也少，你也可以在本项目基础上改用其他模型来玩玩。