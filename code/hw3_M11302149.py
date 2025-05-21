import numpy as np
import scipy.io.wavfile as wavfile

# 定義和弦對應的簡譜音符
chord_dict = {
    'c': [1, 3, 5],     # C major chord (C E G)
    'f': [4, 6, 1],     # F major chord (F A C)
    'g': [5, 7, 2],     # G major chord (G B D)
    'am': [6, 1, 3],    # A minor chord (A C E)
    'dm': [2, 4, 6],    # D minor chord (D F A)
    'em': [3, 5, 7],    # E minor chord (E G B)
}

def getmusic_with_chords(melody, chords, beats, name, bpm=120, tone=1):
    # 定義取樣率
    fs = 44100  # 44.1 kHz 標準取樣率

    # 定義簡譜對應的基本頻率（C大調）
    freq_dict = {
        1: 261.63,  # C4
        2: 293.66,  # D4
        3: 329.63,  # E4
        4: 349.23,  # F4
        5: 392.00,  # G4
        6: 440.00,  # A4
        7: 493.88   # B4
    }

    # 根據BPM計算一拍的持續時間
    base_beat_duration = 60 / bpm

    # 定義音符間的空隙時間（秒）
    gap_duration = 0.02  # 20毫秒小空隙
    gap = np.zeros(int(fs * gap_duration), dtype=np.float32)

    # 初始化整首歌曲的波形
    song = np.array([], dtype=np.float32)

    for m, c, b in zip(melody, chords, beats):
        duration = base_beat_duration * b
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        # 主旋律部分
        freq_melody = freq_dict.get(m, 261.63)
        main_note = generate_tone_wave(freq_melody, t, tone)

        # 和弦部分（若有）
        if c != 0:
            chord_notes = chord_dict.get(c, [])
            chord_wave = np.zeros_like(t)
            for note_c in chord_notes:
                freq_c = freq_dict.get(note_c, 261.63)
                chord_wave += generate_tone_wave(freq_c, t, tone)
            chord_wave = chord_wave / len(chord_notes)
            chord_wave = 0.5 * chord_wave  # 和弦音量降低一半
            note = main_note + chord_wave
        else:
            note = main_note

        # 正規化單個音的音量（避免超過範圍）
        note = note / np.max(np.abs(note))

        # 全段線性下降（模擬真實樂器自然衰減）
        fade = np.linspace(1, 0, len(note))
        note *= fade

        # 將音符和空隙串接到整首歌曲
        song = np.concatenate((song, note, gap))

    # 將波形正規化到 int16 範圍
    song = song * (2**15 - 1)
    song = song.astype(np.int16)

    # 儲存歌曲成 .wav 檔案
    wavfile.write(f"{name}.wav", fs, song)

    print(f"音樂檔 '{name}.wav' 生成完成！")

def generate_tone_wave(frequency, t, tone):
    # 根據tone設定不同音色
    if tone == 1:
        # 鋼琴音色（基頻+2倍頻+3倍頻）
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        wave += 0.2 * np.sin(2 * np.pi * 2 * frequency * t)
        wave += 0.1 * np.sin(2 * np.pi * 3 * frequency * t)
        wave = wave / (0.5 + 0.2 + 0.1)
    elif tone == 2:
        # 吉他音色（基頻+2倍頻+快速衰減）
        wave = 0.6 * np.sin(2 * np.pi * frequency * t)
        wave += 0.2 * np.sin(2 * np.pi * 2 * frequency * t)
        wave += 0.1 * np.sin(2 * np.pi * 3 * frequency * t)
        envelope = np.exp(-4 * t)
        wave *= envelope
        wave = wave / (0.6 + 0.2 + 0.1)
    elif tone == 3:
        # 鐘聲音色（加高次倍頻）
        wave = 0.5 * np.sin(2 * np.pi * frequency * t)
        wave += 0.3 * np.sin(2 * np.pi * 2.5 * frequency * t)
        wave += 0.2 * np.sin(2 * np.pi * 4 * frequency * t)
        wave = wave / (0.5 + 0.3 + 0.2)
    else:
        # 預設鋼琴音色
        wave = np.sin(2 * np.pi * frequency * t)
    return wave

# 範例使用方式
if __name__ == "__main__":
    melody = [1, 1, 5, 5, 6, 6, 5]   # 主旋律音符
    chords = ['c', 0, 0, 0, 'f', 0, 'c']  # 和弦（0代表無）
    beats = [1, 1, 1, 1, 1, 1, 2]    # 每個音的拍子
    name = 'twinkle_with_chords'     # 輸出檔案名稱
    bpm = 130                        # 設定BPM
    tone = 1                        # 音色選擇（1:鋼琴, 2:吉他, 3:鐘聲）
    getmusic_with_chords(melody, chords, beats, name, bpm, tone)




