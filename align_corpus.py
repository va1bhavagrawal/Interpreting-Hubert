import os 
import os.path as osp 
import shutil 

ROOT_WAV_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/wav48"
ROOT_TXT_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/txt" 
ROOT_CORPUS_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/corpuses"
ROOT_TEXTGRID_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/aligned_corpuses" 

SPEAKERS = ["p300"] 

for speaker in SPEAKERS: 
    wav_dir = osp.join(ROOT_WAV_DIR, speaker) 
    assert osp.exists(wav_dir), f"{wav_dir = }" 
    txt_dir = osp.join(ROOT_TXT_DIR, speaker) 
    assert osp.exists(txt_dir), f"{txt_dir = }" 
    corpus_dir = osp.join(ROOT_CORPUS_DIR, speaker) 
    aligned_corpus_dir = osp.join(ROOT_TEXTGRID_DIR, speaker) 
    os.makedirs(corpus_dir, exist_ok=True)  
    for wav_file in os.listdir(wav_dir): 
        if wav_file.find("wav") == -1: 
            continue 
        wav_path = osp.join(wav_dir, wav_file) 
        new_wav_path = osp.join(corpus_dir, wav_file) 
        shutil.copy(wav_path, new_wav_path)  
        txt_path = osp.join(txt_dir, wav_file.replace("wav", "txt")) 
        new_txt_path = new_wav_path.replace("wav", "txt") 
        shutil.copy(txt_path, new_txt_path) 

    os.system(f"mfa align {corpus_dir} english_us_arpa english_us_arpa {aligned_corpus_dir}") 