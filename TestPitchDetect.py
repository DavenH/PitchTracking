import math
import os
from typing import List

import librosa
import librosa.display
import torch
import torchaudio
import torchaudio.functional as taf
import torch.nn.functional as tnnf
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

import PitchDetector as pd
from PitchDetector import PitchFrame, SaveAction


class AnimAction(SaveAction):
    def __init__(self, audio, sample_rate, dest_name, dest_folder,
                 fps=5, bitrate=6000, interval=250, repeat_delay=3000):
        self.img_frames = []
        self.audio, self.sample_rate = (audio, sample_rate)
        self.dest_name, self.dest_folder = (dest_name, dest_folder)
        self.fps, self.bitrate, self.interval, self.repeat_delay = (fps, bitrate, interval, repeat_delay)
        self.fig, _ = plt.subplots(2, 2, figsize=(16, 9))

        grid = gridspec.GridSpec(2, 2, height_ratios=[5, 1])
        ax0 = plt.subplot(grid[0, 0])
        ax1 = plt.subplot(grid[1, :])
        ax2 = plt.subplot(grid[0, 1])
        plt.subplots_adjust(left=0.03, bottom=0.05, right=1, top=1, wspace=0, hspace=0)
        self.ax = (ax0, ax1, ax2)

    def on_iteration(self, pitch_iter_args, index: int):
        cycle_frames, correlogram, cyclogram, shifts, total_adjustments = pitch_iter_args

        ax0, ax1, ax2 = self.ax
        spread = max(1, round(correlogram.size(1) / 700))

        for j in range(correlogram.size(0)):
            correlogram[j, shifts[j] - spread:shifts[j] + spread] = 1

        corr_im = ax0.pcolormesh(correlogram.t().cpu(), cmap="magma")
        audio_im = librosa.display.waveplot(self.audio.cpu().numpy(), self.sample_rate, ax=ax1, color="gray")
        xmin, xmax = ax1.get_xbound()

        # plot the period boundaries over the waveform
        lines = []
        for frame in cycle_frames:
            x = (frame.offset / self.audio.size(0)) * (xmax - xmin)
            lines.append([(x, -0.03), (x, 0.03)])

        col = LineCollection(lines, linewidths=1., colors="black")
        lines_im = ax1.add_collection(col)

        ragged_im = ax2.pcolormesh(cyclogram.t().cpu(), cmap="icefire", animated=True)
        text = ax2.annotate(f"{index}, {total_adjustments:.2f}", (2, 2))

        self.img_frames.append([text, ragged_im, corr_im, audio_im, lines_im])

    def finish(self, error_pct: str = ""):
        print(f"Saving animation of {len(self.img_frames)} frames")

        ffmpeg_writer = animation.writers['ffmpeg']
        writer = ffmpeg_writer(fps=self.fps, metadata=dict(artist='Me'), bitrate=self.bitrate)
        ani = animation.ArtistAnimation(self.fig, self.img_frames, interval=self.interval,
                                        repeat_delay=self.repeat_delay, blit=False)
        os.makedirs(self.dest_folder, exist_ok=True)
        ani.save(f"{self.dest_folder}/{self.dest_name}-{error_pct}.mp4", writer=writer)
        plt.close(self.fig)


class LagAnimAction(AnimAction):
    def __init__(self, audio, sample_rate, dest_name, dest_folder):
        super().__init__(audio, sample_rate, dest_name, dest_folder, 20, 5000, 100, 1000)
        self.min_lag, self.max_lag = (10, 10)
        self.y_scale = 1
        self.rshp_im = self.max_im = self.res_im = self.res_im = self.text_im = None

    def start(self, min_lag, max_lag):
        self.min_lag, self.max_lag = (min_lag, max_lag)
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(2, 1, figsize=(16, 9))

        self.y_scale = self._get_fig_h() / 2 / (max_lag - min_lag + 1)

    def on_pitch_est_iteration(self, args, i):
        ax0, ax1 = self.ax
        rshp, resogram, indices = args
        self.rshp_im = ax0.imshow(rshp.t().cpu(), cmap="icefire", aspect='auto',
                                  extent=(0, 16, 0, 9), interpolation='nearest', animated=True)

        fig_w = self.fig.bbox.width
        fig_h = self.fig.bbox.height
        self.res_im = ax1.imshow(resogram.cpu(), cmap="icefire", extent=(0, fig_w, fig_h / 2, 0), animated=True)

        x = [fig_w * i / (indices.size(0) - 1) for i in range(indices.size(0))]
        y = [self.y_scale * val.item() for val in indices]
        self.max_im, = ax1.plot(x, y, color="black")

        self.text_im = ax0.annotate(f"lag: {i}", (1, 1))
        self.img_frames.append([self.text_im, self.rshp_im, self.res_im, self.max_im])

    def post_process(self, frames):
        self._update_line_image(frames, n_frames=4)

        for j in range(3):
            frames = pd.repair_frames(frames, self.min_lag, self.max_lag, True)
            self._update_line_image(frames, n_frames=10)

        frames = pd.interp_frames_by_window(frames, 0, min_period=self.min_lag, max_period=self.max_lag)
        self._update_line_image(frames, n_frames=10)

        return frames

    def on_iteration(self, pitch_iter_args, i):
        y_span = self.max_lag - self.min_lag + 1
        cycle_frames, correlogram, cyclogram, shifts, total_adjustments = pitch_iter_args
        limited_frames = [limit(frame, self.audio.size(0), y_span) for frame in cycle_frames]

        self.rshp_im = self.ax[0].imshow(cyclogram.t().flip(dims=[0]).cpu(), cmap="icefire",
                                         aspect='auto', extent=(0, 16, 0, 9),
                                         interpolation='nearest', animated=True)
        self._update_line_image(limited_frames, n_frames=4)

    def finish(self, error_pct: str = ""):
        self._repeat_last_frame(40)
        super().finish(error_pct)

    def _get_fig_h(self):
        return self.fig.bbox.height

    def _repeat_last_frame(self, times:int = 10):
        self.img_frames += [self.img_frames[-1] for _ in range(times)]

    def _update_line_image(self, frames, n_frames: int = 10):
        self.max_im = self._make_image(frames)
        self.img_frames.append([self.text_im, self.rshp_im, self.res_im, self.max_im])
        self._repeat_last_frame(n_frames)

    def _make_image(self, frames):
        axis, img_width, y_scale, y_offset = (self.ax[1], self.fig.bbox.width, self.y_scale, self.min_lag)

        scale = img_width / frames[-1].offset
        x = [frame.offset * scale for frame in frames]
        y = [(frame.period - y_offset) * y_scale for frame in frames]

        x.append(img_width)
        y.append(y[-1])
        img, = axis.plot(x, y, color="black")

        return img


class DummyLagSaveAction(SaveAction):
    def start(self, min_lag, max_lag):
        self.min_lag, self.max_lag = (min_lag, max_lag)

    def post_process(self, frames):
        for j in range(3):
            frames = pd.repair_frames(frames, self.min_lag, self.max_lag, True)

        frames = pd.interp_frames_by_window(frames, 0, min_period=self.min_lag, max_period=self.max_lag)

        return frames

    def on_pitch_est_iteration(self, args, i):
        pass


def make_cyclogram(audio: torch.Tensor, frames: List[PitchFrame]) -> torch.Tensor:
    max_span = max(frames, key=lambda fr: fr.period)
    height = int(1.5 * (max_span.period + 1))

    with torch.no_grad():
        window = torch.ones(height, device=audio.device)
        quarter = round(height / 4)
        x = torch.linspace(start=-math.pi / 2, end=math.pi / 2, steps=quarter)

        window[0:quarter] = (torch.sin(x) + 1) * 0.5
        window.mul_(window.flip(dims=[0]))
        cyclogram = torch.zeros(len(frames), height, device=audio.device)

        idx = 0
        for f in frames:
            # expand range so 1/4 of the previous cycle is visible above and 1/4 below; this
            # ensures continuity between cycles, say if this image is used to train a GAN
            start_sub_quarter = f.offset - f.period / 4
            end_add_quarter = f.offset + f.period * 5/4
            segment = audio[round(start_sub_quarter):round(end_add_quarter)]
            if segment.size(0) > 1:
                cyclogram[idx] = tnnf.interpolate(segment.view(1, 1, -1), cyclogram.size(1))
            idx += 1

        # taper out those 1/4 overlap parts for visual clarity
        cyclogram.mul_(window)

    return cyclogram


def plot_waveform_and_spect(signal: torch.Tensor, sample_rate: int,
                            fmin: float, fmax: float,
                            lookahead_cycles: int,
                            frame_length: int, hop_length: int,
                            method: str, file_name: str):
    with torch.no_grad():
        spec_module = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=64)
        spec_module.to(signal.device)
        torch_stft = spec_module.forward(signal)[1:]

        n_harmonics = 256
        stft_db = taf.amplitude_to_DB(x=torch_stft[0:n_harmonics, :], multiplier=5., amin=1e-10,
                                      db_multiplier=math.log10(max(1e-10, 1.)), top_db=30)
        fig, axs = plt.subplots(2, 1, figsize=(16, 9))
        plt.style.use('dark_background')

        folder, name = get_source_name(file_name)
        save_action = AnimAction(signal, sample_rate, name, folder)
        # save_action = DummyAction()

        cycle_frames, correlogram, shifts, ragged, filtered, error_pct = \
            pd.detect_pitch(signal, sample_rate, save_action, fmin, fmax, frame_length=frame_length,
                            hop_length=hop_length, lookahead_cycles=lookahead_cycles, method=method)

        max_period = max(cycle_frames, key=lambda frame: frame.period).period
        max_period = int(max_period + 1)

        spread = round(correlogram.size(1) / 700)
        for i in range(correlogram.size(0)):
            correlogram[i, shifts[i] - spread:shifts[i] + spread] = 1

        axs[0].pcolormesh(stft_db.cpu(), cmap="afmhot")
        axs[1].pcolormesh(ragged[:, 0:max_period].t().cpu(), cmap="icefire")

        # cyclogram = make_cyclogram(signal, cycle_frames)
        # axs[0, 1].set_yscale('symlog')
        # gold_ragged, _ = pd.make_ragged_cyclogram(signal, cycle_frames)
        # axs[0, 1].pcolormesh(gold_ragged.t().cpu(), cmap="icefire")
        # axs[0, 0].pcolormesh(correlogram.t().cpu(), cmap="magma")
        # axs[1, 1].pcolormesh(cyclogram.t().cpu(), cmap="icefire")

        plt.subplots_adjust(left=0.05, bottom=0, right=1, top=1, wspace=0, hspace=0)
        new_dir = f"media/{folder}/"
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        fname = new_dir + f"{name}-{error_pct}.jpg"

        plt.savefig(fname)
        plt.show()

def limit(frame: PitchFrame, high_x, high_y):
    return PitchFrame(max(0, min(high_x, frame.offset)), min(high_y, frame.period), frame.confidence, 0)

def get_source_name(file_name: str):
    fname = file_name.split("/")
    folder = fname[-2]
    name = fname[-1].split(".")[0].replace(" ", "_")
    return folder, name


def run_test(file_name: str, sample_offsets: List = (0, 200000)):
    device = torch.device("cuda:0")

    waveform, sample_rate = torchaudio.load(file_name)
    waveform = waveform.to(device)
    signal = waveform[0, sample_offsets[0]:sample_offsets[0] + sample_offsets[1]]

    plot_waveform_and_spect(signal, sample_rate, hop_length=256, frame_length=1024,
                            lookahead_cycles=3, fmin=40., fmax=800., method='crepe',
                            file_name=file_name)
