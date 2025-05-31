import os
import io
import sys
from pydub import AudioSegment
from pydub.playback import play
import torch
import numpy as np
from scipy.io import wavfile

class Everything(str):
    def __ne__(self, __value: object) -> bool:
        return False

class ailab_PlayAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "AUDIO": ("AUDIO", {"forceInput": True}),
                "audio_path": ("STRING", {"default": ""}),
                "autoplay": ("BOOLEAN", {"default": True, "label": "Auto Play"})
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("AUDIO", "AUDIO_PATH",)
    FUNCTION = "execute"
    CATEGORY = "🧪AILab/🌸Pollinations"
    OUTPUT_NODE = True
    ALWAYS_CHANGED = True  # This tells ComfyUI to always execute this node
    
    def play_audio(self, AUDIO=None, audio_path=None, autoplay=True):
        try:
            sound = None
            path = ""
            
            if AUDIO is not None:
                if isinstance(AUDIO, dict) and 'waveform' in AUDIO:
                    waveform = AUDIO['waveform']
                    sample_rate = AUDIO.get('sample_rate', 44100)
                    
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.cpu().numpy()
                    
                    if waveform.dtype.kind == 'f':
                        waveform = (waveform * 32767).astype(np.int16)
                    
                    temp_wav = io.BytesIO()
                    wavfile.write(temp_wav, sample_rate, waveform)
                    temp_wav.seek(0)
                    sound = AudioSegment.from_wav(temp_wav)
                    path = ""
                
                elif isinstance(AUDIO, AudioSegment):
                    sound = AUDIO
                    path = ""
                else:
                    raise ValueError(f"Unsupported AUDIO type: {type(AUDIO)}")
            
            elif audio_path and os.path.exists(audio_path):
                sound = AudioSegment.from_file(audio_path)
                path = audio_path
            
            if autoplay and sound is not None:
                if sys.platform.startswith('win'):
                    wav_io = io.BytesIO()
                    sound.export(wav_io, format='wav')
                    wav_data = wav_io.getvalue()
                    import winsound
                    winsound.PlaySound(wav_data, winsound.SND_MEMORY)
                else:
                    play(sound)
            
            return sound, path
                
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return None, ""

    def execute(self, AUDIO=None, audio_path=None, autoplay=True):
        sound, path = self.play_audio(AUDIO, audio_path, autoplay)
        return (sound, path)
    
NODE_CLASS_MAPPINGS = {
    "ailab_PlayAudio": ailab_PlayAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ailab_PlayAudio": "Play Audio 🔊",
} 