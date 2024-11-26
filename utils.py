import os 
import os.path as osp 

import numpy as np

def parse_textgrid(file_path):
    """
    Parse the TextGrid file and extract the phoneme intervals.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    phoneme_intervals = []
    is_phone_tier = False
    started_intervals = False 
    
    for line in lines:
        line = line.strip()
        if 'name = "phones"' in line:
            print(f"Found phones tier!") 
            is_phone_tier = True
        if is_phone_tier and line.find("intervals [") != -1:
            interval = {}
            started_intervals = True
        if started_intervals and is_phone_tier and line.startswith("xmin ="):
            interval["xmin"] = float(line.split("=")[1].strip())
        if started_intervals and is_phone_tier and line.startswith("xmax ="):
            interval["xmax"] = float(line.split("=")[1].strip())
        if started_intervals and is_phone_tier and line.startswith('text ='):
            interval["text"] = line.split("=")[1].strip().strip('"')
            phoneme_intervals.append(interval)
        if started_intervals and is_phone_tier and line.startswith("item ["): 
            break  # End of phone tier
    
    return phoneme_intervals

def generate_phoneme_ground_truth(phoneme_intervals, frame_size, hop_size, total_duration):
    """
    Generate the phoneme ground truth for frames of the given size and hop.
    """
    print(f"{total_duration = }")
    print(f"{hop_size = }")
    frame_starts = np.arange(0, total_duration, hop_size)
    print(f"{len(frame_starts) = }")
    print(f"{frame_starts = }")
    frame_ends = frame_starts + frame_size
    ground_truth = []

    for start, end in zip(frame_starts, frame_ends):
        matched_phoneme = None
        for interval in phoneme_intervals:
            if start < interval["xmax"] and end > interval["xmin"]:
                matched_phoneme = interval["text"]
                break
        ground_truth.append(matched_phoneme if matched_phoneme else "")

    return ground_truth

# Example usage:
if __name__ == "__main__": 
    file_path = "./pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/aligned_corpuses/p376/p376_150.TextGrid"  # Replace with your file path
    assert osp.exists(file_path)
    frame_size = 0.025  # 20 ms
    hop_size = 0.020  # 25 ms

    # Parse the TextGrid and compute ground truth
    phoneme_intervals = parse_textgrid(file_path)
    total_duration = max(interval["xmax"] for interval in phoneme_intervals)  # Determine xmax from the intervals
    phoneme_ground_truth = generate_phoneme_ground_truth(phoneme_intervals, frame_size, hop_size, total_duration)

    # Print the frame-wise phoneme ground truth
    for i, phoneme in enumerate(phoneme_ground_truth):
        print(f"Frame {i}: {phoneme}")
