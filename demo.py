import librosa
import numpy as np
import argparse
from scipy.signal import savgol_filter,butter
import torch
from model import EmoTalk
import random
import os, subprocess
import shlex
import time


@torch.no_grad()
def test(args):
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    eye1 = np.array([0.36537236, 0.950235724, 0.95593375, 0.916715622, 0.367256105, 0.119113259, 0.025357503])
    eye2 = np.array([0.234776169, 0.909951985, 0.944758058, 0.777862132, 0.191071674, 0.235437036, 0.089163929])
    eye3 = np.array([0.870040774, 0.949833691, 0.949418545, 0.695911646, 0.191071674, 0.072576277, 0.007108896])
    eye4 = np.array([0.000307991, 0.556701422, 0.952656746, 0.942345619, 0.425857186, 0.148335218, 0.017659493])
    model = EmoTalk(args)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)), strict=False)
    model = model.to(args.device)
    model.eval()
    wav_path = args.wav_path
    file_name = wav_path.split('/')[-1].split('.')[0]
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    audio = torch.FloatTensor(speech_array).unsqueeze(0).to(args.device)
    level = torch.tensor([1]).to(args.device)
    person = torch.tensor([0]).to(args.device)
    prediction = model.predict(audio, level, person)
    prediction = prediction.squeeze().detach().cpu().numpy()

    if args.post_processing:
        output = np.zeros((prediction.shape[0], prediction.shape[1]))
        for i in range(prediction.shape[1]):
            if 0 <= i <= 19:
                output[:, i] = adaptive_smooth(prediction[:, i], eps=0.3, alpha=0.8)
            else:
                output[:, i] = savgol_filter(prediction[:, i], 5, 2)

        output[:, 8] = 0
        output[:, 9] = 0
        i = random.randint(0, 60)
        while i < output.shape[0] - 7:
            eye_num = random.randint(1, 4)
            if eye_num == 1:
                output[i:i + 7, 8] = eye1
                output[i:i + 7, 9] = eye1
            elif eye_num == 2:
                output[i:i + 7, 8] = eye2
                output[i:i + 7, 9] = eye2
            elif eye_num == 3:
                output[i:i + 7, 8] = eye3
                output[i:i + 7, 9] = eye3
            else:
                output[i:i + 7, 8] = eye4
                output[i:i + 7, 9] = eye4
            time1 = random.randint(60, 180)
            i = i + time1
        np.save(os.path.join(result_path, "{}.npy".format(file_name)), output)  # with postprocessing (smoothing and blinking)
    else:
        np.save(os.path.join(result_path, "{}.npy".format(file_name)), prediction)  # without post-processing

def adaptive_smooth(channel: np.ndarray, eps=0.02, alpha=0.8):
    sm = channel.copy()
    for t in range(1, len(channel)):
        delta = channel[t] - sm[t-1]
        if abs(delta) < eps:
            sm[t] = alpha*sm[t-1] + (1-alpha)*channel[t]
        else:
            sm[t] = channel[t]
    return sm


def bilateral_filter_1d(x, sigma_time=2, sigma_val=0.05, window=5):
    T = len(x)
    half = window // 2
    y = np.zeros_like(x)
    # 시간 가중치 미리 계산
    times = np.arange(-half, half+1)
    w_time = np.exp(-0.5*(times/sigma_time)**2)
    
    for t in range(T):
        # 윈도우 인덱스
        idxs = np.clip(np.arange(t-half, t+half+1), 0, T-1)
        vals = x[idxs]
        # 값 차이 가중치
        w_val = np.exp(-0.5*((vals - x[t]) / sigma_val)**2)
        # 합성 가중치
        w = w_time * w_val
        # 가중합
        y[t] = np.sum(w * vals) / np.sum(w)
    return y

def render_video(args):
    wav_name = args.wav_path.split('/')[-1].split('.')[0]
    image_path = os.path.join(args.result_path, wav_name)
    os.makedirs(image_path, exist_ok=True)
    image_temp = image_path + "/%d.png"
    output_path = os.path.join(args.result_path, wav_name + ".mp4")
    blender_path = args.blender_path
    python_path = "./render.py"
    blend_path = "./render.blend"
    cmd = '{} -t 64 -b {} -P {} -- "{}" "{}" '.format(blender_path, blend_path, python_path, args.result_path, wav_name)
    cmd = shlex.split(cmd)
    p = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print('[{}]'.format(line))
    if p.returncode == 0:
        print('Subprogram success')
    else:
        print('Subprogram failed')

    cmd = 'ffmpeg -r 30 -i "{}" -i "{}" -pix_fmt yuv420p -s 512x768 "{}" -y'.format(image_temp, args.wav_path, output_path)
    subprocess.call(cmd, shell=True)

    cmd = 'rm -rf "{}"'.format(image_path)
    subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(
        description='EmoTalk: Speech-driven Emotional Disentanglement for 3D Face Animation')
    parser.add_argument("--wav_path", type=str, default="./audio/angry1.wav", help='path of the test data')
    parser.add_argument("--bs_dim", type=int, default=52, help='number of blendshapes:52')
    parser.add_argument("--feature_dim", type=int, default=832, help='number of feature dim')
    parser.add_argument("--period", type=int, default=30, help='number of period')
    parser.add_argument("--device", type=str, default="cuda", help='device')
    parser.add_argument("--model_path", type=str, default="./pretrain_model/EmoTalk.pth",
                        help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="./result/", help='path of the result')
    parser.add_argument("--max_seq_len", type=int, default=5000, help='max sequence length')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--post_processing", type=bool, default=True, help='whether to use post processing')
    parser.add_argument("--blender_path", type=str, default="./blender/blender", help='path of blender')
    args = parser.parse_args()

    t0 = time.perf_counter()
    test(args)
    print(f"test func: {time.perf_counter()-t0:.2f}s", flush=True)
    
    t1 = time.perf_counter()
    render_video(args)
    print(f"video render: {time.perf_counter()-t1:.2f}s", flush=True)


if __name__ == "__main__":
    main()
