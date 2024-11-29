import os
import os.path as osp
import shutil
import argparse

def main(root_wav_dir, root_txt_dir, root_corpus_dir, root_textgrid_dir, speakers):
    for speaker in speakers:
        wav_dir = osp.join(root_wav_dir, speaker)
        assert osp.exists(wav_dir), f"{wav_dir = }"
        txt_dir = osp.join(root_txt_dir, speaker)
        assert osp.exists(txt_dir), f"{txt_dir = }"
        corpus_dir = osp.join(root_corpus_dir, speaker)
        aligned_corpus_dir = osp.join(root_textgrid_dir, speaker)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and align audio files for MFA.")
    parser.add_argument("--root_wav_dir", type=str, required=True, help="Path to the root wav directory.")
    parser.add_argument("--root_txt_dir", type=str, required=True, help="Path to the root txt directory.")
    parser.add_argument("--root_corpus_dir", type=str, required=True, help="Path to the root corpus directory.")
    parser.add_argument("--root_textgrid_dir", type=str, required=True, help="Path to the root aligned TextGrid directory.")
    parser.add_argument("--speakers", nargs="+", required=True, help="List of speaker IDs to process.")

    args = parser.parse_args()
    main(args.root_wav_dir, args.root_txt_dir, args.root_corpus_dir, args.root_textgrid_dir, args.speakers)



# import os 
# import os.path as osp 
# import shutil 

# ROOT_WAV_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/wav48"
# ROOT_TXT_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/txt" 
# ROOT_CORPUS_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/corpuses"
# ROOT_TEXTGRID_DIR = "/data/vaibhav/segment_analysis/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/aligned_corpuses" 

# SPEAKERS = ["p300"] 

# for speaker in SPEAKERS: 
#     wav_dir = osp.join(ROOT_WAV_DIR, speaker) 
#     assert osp.exists(wav_dir), f"{wav_dir = }" 
#     txt_dir = osp.join(ROOT_TXT_DIR, speaker) 
#     assert osp.exists(txt_dir), f"{txt_dir = }" 
#     corpus_dir = osp.join(ROOT_CORPUS_DIR, speaker) 
#     aligned_corpus_dir = osp.join(ROOT_TEXTGRID_DIR, speaker) 
#     os.makedirs(corpus_dir, exist_ok=True)  
#     for wav_file in os.listdir(wav_dir): 
#         if wav_file.find("wav") == -1: 
#             continue 
#         wav_path = osp.join(wav_dir, wav_file) 
#         new_wav_path = osp.join(corpus_dir, wav_file) 
#         shutil.copy(wav_path, new_wav_path)  
#         txt_path = osp.join(txt_dir, wav_file.replace("wav", "txt")) 
#         new_txt_path = new_wav_path.replace("wav", "txt") 
#         shutil.copy(txt_path, new_txt_path) 

#     os.system(f"mfa align {corpus_dir} english_us_arpa english_us_arpa {aligned_corpus_dir}") 