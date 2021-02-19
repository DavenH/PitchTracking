import copy
from typing import List, Optional

import torch.nn.functional as tnnf
import torchaudio.functional as taf
import torch
import math
import numpy as np
import torch.fft
import librosa


class PitchFrame:
    def __init__(self, offset: int, period: float, confidence: float, trough_idx: int):
        self.offset: int = offset
        self.period: float = period
        self.confidence: float = confidence
        self.tonality: float = 1
        self.trough_idx = trough_idx

    def is_atonal(self, confidence_thres: float = 0.01) -> bool:
        return self.period == 0. or math.isnan(self.period) or self.confidence < confidence_thres

    def end(self):
        return self.offset + self.period

    def __repr__(self):
        return f"{self.offset} {self.period:0.2f}"


class FrequencyBin:
    def __init__(self, average_period):
        self.population: int = 1
        self.average_period: float = average_period

    def accumulate(self, period):
        self.average_period = (self.population * self.average_period + period) / (self.population + 1)
        self.population += 1

    def contains(self, period, tolerance: float = 0.2):
        return (1 - tolerance) * self.average_period <= period <= (1 + tolerance) * self.average_period

    def __repr__(self):
        return f"{self.average_period}-{self.population}"


class SaveAction:
    def on_iteration(self, args: (), i: int):
        pass

    def finish(self, error_pct: str = ""):
        pass


def detect_pitch(audio: torch.Tensor,
                 sample_rate: int,
                 save_action: SaveAction,
                 fmin: float = 30,
                 fmax: float = 800,
                 frame_length: int = 512,
                 hop_length: int = 128,
                 method: str = 'crepe',
                 lookahead_cycles: int = 1,
                 cross_corr_iters: int = 200,
                 max_iters_no_improve: int = 20):
    audio = taf.highpass_biquad(audio, sample_rate, cutoff_freq=max(10., fmin - 15))

    min_period = sample_rate / (fmax * 1.)
    max_period = sample_rate / (fmin * 0.8)

    if method == 'pyin':
        np_audio = audio.cpu().numpy()
        pitch_env_np, voiced, voiced_prob = librosa.pyin(np_audio,
                                                         fmin=fmin, fmax=fmax,
                                                         sr=sample_rate,
                                                         hop_length=hop_length,
                                                         frame_length=frame_length)

    elif method == 'crepe':
        pitch_env_np, voiced_prob = pitch_detect_crepe(audio, sample_rate,
                                                       hop_length=hop_length,
                                                       fmin=fmin, fmax=fmax)
    # elif method == 'lag':
    #     pass
    else:
        raise Exception("Unknown detection method ", method)

    frames = make_frames_equal_spaced(pitch_env_np, voiced_prob, sample_rate, hop_length)

    if len(frames) == 0:
        raise Exception(f"No good frames in pitch estimation. Parameters are: {fmin}, {fmax}")

    bins = fill_frequency_bins(frames)
    mode_period = max(bins, key=lambda k: k.population).average_period

    cycle_frames = filter_atonal_frames(frames, frame_length, min_period, max_period, 0.02)
    cycle_frames = repair_frames(cycle_frames, min_period, max_period, True)
    cycle_frames = interp_frames_by_window(cycle_frames, frame_length, min_period, max_period)

    filtered = taf.lowpass_biquad(audio, sample_rate, cutoff_freq=2000, Q=0.7)

    correlogram = shifts = None
    best_iter = iters_without_imprv = 0
    shift_limit = mode_period / 3
    best_total_adj = 1e10

    for i in range(cross_corr_iters):
        if round(shift_limit) == 0:
            break

        args = make_crosscorr_adjustments(filtered, cycle_frames, min_period, max_period,
                                          lookahead_cycles, round(shift_limit))

        total_adjustments = args[-1]
        save_action.on_iteration(args, i)

        best_total_adj, iters_without_imprv, best_iter = \
            test_if_best_iter(total_adjustments, i, best_iter,
                              best_total_adj, iters_without_imprv)

        if iters_without_imprv > max_iters_no_improve:
            break

        print(f"{i}\t{total_adjustments:.2f}\t{shift_limit:.3f}")
        shift_limit = min(shift_limit * 0.85 + 0.08, mode_period / 5)

    cyclogram = make_ragged_cyclogram(audio, cycle_frames, True)

    error_pct = f"{best_total_adj:.2f}"
    save_action.finish(error_pct)

    print(f"Iteration {best_iter} had most corr: {error_pct}")
    return cycle_frames, correlogram, shifts, cyclogram, audio, error_pct


def test_if_best_iter(error, itr, best_iter, best_total_shifts, iters_without_improvement):
    if error < best_total_shifts:
        best_total_shifts = error
        best_iter = itr
        iters_without_improvement = 0
    else:
        iters_without_improvement += 1

    return best_total_shifts, iters_without_improvement, best_iter


def pitch_detect_lag(audio: torch.Tensor,
                     sample_rate: int,
                     save_action,
                     log_scale: bool = False,
                     fmin: float = 40,
                     fmax: float = 800):
    min_lag = round(sample_rate / fmax)
    max_lag = round(sample_rate / fmin)

    save_action.start(min_lag, max_lag)

    length = audio.size(0)
    max_stride = int(math.floor(length / min_lag))

    with torch.no_grad():
        resogram = torch.zeros(max_lag - min_lag + 1, max_stride - 1, device=audio.device)
        sums = torch.zeros(max_lag - min_lag + 1, max_stride - 1, device=audio.device)

        import time
        start_time = time.time()
        for lag in range(min_lag, max_lag + 1):
            stride = int(math.floor(length / lag))
            rshp_len = int(stride * lag)

            offset = audio.size(0) - 1 - rshp_len
            rshp = audio[0:-offset - 1].view(stride, -1)

            # avoid octaving errors by scaling the sum-of-differences as a function of lag
            curr_sum = torch.sum(torch.abs(rshp[0:-1] - rshp[1:]) / math.pow(lag, 0.4), dim=1)

            # stretch this vector to a constant -- max_stride-1 -- to map
            # sum vectors with varying size to the same time interval
            interp_sum = tnnf.interpolate(curr_sum.view(1, 1, -1), max_stride - 1,
                                          mode='linear', align_corners=False)

            sums[lag - min_lag, :] = interp_sum
            diff_sum_exp = torch.exp(- interp_sum)
            abs_sum = torch.sum(torch.abs(rshp[0:-1] + rshp[1:]), dim=1)
            interp_abs = tnnf.interpolate(abs_sum.view(1, 1, -1), max_stride - 1,
                                          mode='linear', align_corners=False)
            prod = diff_sum_exp * interp_abs

            if log_scale:
                sign = torch.sign(diff_sum_exp)
                resonances = sign * torch.log1p(0.0001 + 5 * sign * prod)  # / math.sqrt(i)
            else:
                resonances = prod  # / math.sqrt(i)

            resogram[lag - min_lag, :] = resonances
            max_val, indices = resogram.max(dim=0)

            save_action.on_pitch_est_iteration((rshp, resogram, indices), lag)

        end_time = time.time()
        print(f"Calculation took {end_time - start_time:0.4f} seconds")

        max_val, indices = resogram.max(dim=0)
        max_val /= max_val.max()

        frames = []
        for lag, (val, idx) in enumerate(zip(max_val, indices)):
            frames.append(PitchFrame(round(lag / (max_stride - 1) * audio.size(0)),
                                     idx.item() + min_lag, val.item(), -1))

        frame_length = round(int(audio.size(0)) / (max_stride - 1))
        frames = filter_atonal_frames(frames, frame_length, min_lag, max_lag, 0.02)
        frames = save_action.post_process(frames)

        bins = fill_frequency_bins(frames)
        mode_period = max(bins, key=lambda k: k.population).average_period
        filtered = taf.lowpass_biquad(audio, sample_rate, cutoff_freq=2000, Q=0.7)

        shift_limit = mode_period / 3
        best_total_shifts = 1e10

        for i in range(100):
            if round(shift_limit) == 0:
                break

            args = make_crosscorr_adjustments(filtered, frames, min_lag, max_lag, 3, round(shift_limit))
            save_action.on_iteration(args, i)

            total_adjustments = args[-1]
            best_total_shifts, iters_without_improvement, best_iter = \
                test_if_best_iter(total_adjustments, i, best_iter, best_total_shifts, iters_without_improvement)

            if iters_without_improvement > 10:
                break

            shift_limit = min(shift_limit * 0.85 + 0.08, mode_period / 5)

        save_action.finish()
        start_time = time.time()
        print(f"Finished. Saving took {start_time - end_time:0.4f} seconds")


def make_crosscorr_adjustments(filtered: torch.Tensor,
                               cycle_frames: List[PitchFrame],
                               min_period: float, max_period: float,
                               lookahead: int,
                               shift_limit: int,
                               voiced_filter_limit: float = 0.1
                               ):

    cycle_frames = repair_frames(cycle_frames, min_period, max_period, False, 1.2)
    ragged = make_ragged_cyclogram(filtered, cycle_frames, False)
    mags, cplx, *_ = get_fft_spectrograms(ragged, return_complex=True, return_mags=True)
    num_columns = ragged.size(0)
    conj_vals = torch.conj(cplx)

    # cross-correlate a cycle with its previous cycle
    cross_corr = cplx[1:num_columns] * conj_vals[0:num_columns - 1]
    correlogram = torch.fft.irfft(cross_corr)

    if lookahead > 1:
        cross_corr2 = cplx[2:num_columns] * conj_vals[0:num_columns - 2]
        correlogram2 = torch.fft.irfft(cross_corr2)
        correlogram[0:num_columns - 2] += correlogram2

    if lookahead > 2:
        cross_corr3 = cplx[3:num_columns] * conj_vals[0:num_columns - 3]
        correlogram3 = torch.fft.irfft(cross_corr3)
        correlogram[0:num_columns - 3] += correlogram3

    padded_size = ragged.size(1)

    correlogram = correlogram.roll(padded_size // 2, 1)
    max_vals, shifts_by_index = correlogram.max(dim=1)
    for i in range(correlogram.size(0)):
        correlogram[i] /= (0.5 + max_vals[i])

    abs_rag = ragged.abs()
    max_raw, _ = abs_rag.max(dim=1)

    first_mags = torch.sum(mags[:-1, 0:5], dim=1)
    shifts_by_index_orig = shifts_by_index.clone()
    shifts_by_index -= padded_size // 2

    # filter to only the 'voiced' sections
    shifts_by_index[first_mags < voiced_filter_limit] = 0
    # shifts_by_index[first_mags < voiced_filter_limit] //= 2

    total_shifts = torch.sum(torch.abs(shifts_by_index) * max_raw[0:-1]).item()
    shifts_by_index[shifts_by_index > shift_limit] = shift_limit
    shifts_by_index[shifts_by_index < -shift_limit] = -shift_limit
    cume_shifts = torch.cumsum(shifts_by_index, dim=0)

    new_period_positions = []

    # for (cume_shift, frame) in zip(cume_shifts, cycle_frames[0:-1]):
    #     new_period_positions.append(frame.end() + cume_shift.item())

    # 1. cume_shifts[i] applies to the region around that particular offset of cycle_frames[i].offset
    # 2. The new_offset[i] does not (necessarily) correspond to the cycle_frames[i].offset region
    # 3. y = cycle_frames[i].end() + cume_shifts[i], x = cycle_frames[i].offset;
    #    what is the interpolation of y=f(x) that corresponds to new_offset[j] ?
    cume_itr = frame_itr = 0
    cume_offset = 0.

    while frame_itr < len(cycle_frames) - 1:
        while cume_itr < len(cycle_frames) - 1 and cume_offset < cycle_frames[frame_itr].end():
            end = cycle_frames[cume_itr].end() + cume_shifts[cume_itr].item()
            cume_offset = min(cume_offset + max_period, max(cume_offset + min_period, end))
            new_period_positions.append(cume_offset)
            cume_itr += 1
        frame_itr += 1

    last_period = cycle_frames[-1].end() + cume_shifts[-1].item() - new_period_positions[-1]
    last_period = max(min_period, min(max_period, last_period))

    cume_offset = new_period_positions[-1]

    # ensures we don't have a wandering end to the range on our frames
    while new_period_positions[-1] <= filtered.size(0):
        cume_offset += last_period
        new_period_positions.append(cume_offset)

    cume_offset = 0.
    new_frames = []
    for offset in new_period_positions:
        new_frames.append(PitchFrame(round(cume_offset), offset - cume_offset, 0.5, -1))
        cume_offset += offset - cume_offset

    ragged = make_ragged_cyclogram(filtered, new_frames, True)

    return new_frames, correlogram, ragged, shifts_by_index_orig, total_shifts


def make_ragged_cyclogram(audio: torch.Tensor, frames: List[PitchFrame], mark_ends: bool):
    max_span = max(frames, key=lambda fr: fr.period)
    height = int(1 + max_span.period)
    if not mark_ends:
        next_pow_2 = 2 ** (height - 1).bit_length()
        height = next_pow_2

    cyclogram = torch.zeros(len(frames), height, device=audio.device)
    # cyclogram.fill_(-0.1)

    for i in range(len(frames)):
        f = frames[i]
        end_idx = round(f.end())

        if f.period <= 1:
            continue

        segment = audio[f.offset:end_idx]
        end = segment.size(0)
        cyclogram[i, 0:end] = segment

        if mark_ends and end < height:
            cyclogram[i, end] = 0.5

    return cyclogram


def interp_frames_by_window(frames: List[PitchFrame], frame_length: int,
                            min_period: float = 30, max_period: float = 1500) -> (List, float):
    frames = copy.deepcopy(frames)
    new_frames = []
    last_pos = frames[-1].end()
    cume_offset = 0
    frame_itr = 0

    while cume_offset < last_pos and frame_itr < len(frames):
        y_sum = 0
        conf_sum = 0
        iter_start = frame_itr

        while frame_itr < len(frames) and frames[frame_itr].offset + frame_length / 2 < cume_offset + \
                frames[frame_itr].period:
            y_sum += frames[frame_itr].period
            conf_sum += frames[frame_itr].confidence
            frame_itr += 1

        iter_end = frame_itr
        if iter_end > iter_start:
            size = iter_end - iter_start
            period_avg = max(min_period, min(max_period, y_sum / size))
            frame = PitchFrame(round(cume_offset), period_avg, conf_sum / size, 0)
            cume_offset += period_avg
            new_frames.append(frame)

        else:  # ?
            frame = frames[frame_itr]
            new_frames.append(PitchFrame(round(cume_offset), frame.period, frame.confidence, 0))
            cume_offset += max(min_period, min(max_period, frame.period))

    return new_frames


def get_fft_spectrograms(cyclogram: torch.Tensor,
                         return_mags=False,
                         return_phases=False,
                         return_complex=False):
    cplx = torch.fft.rfft(cyclogram)
    magnitudes = None
    if return_mags:
        magnitudes = torch.abs(cplx)

    phases = None
    if return_phases:
        phases = torch.angle(cplx)

    ret = []
    if return_mags:
        ret.append(magnitudes)
    if return_phases:
        ret.append(phases)
    if return_complex:
        ret.append(cplx)

    return ret


def make_frames_equal_spaced(pitch_array, voiced_prob, sample_rate: int, hop_length: int):
    frames_ = []
    offset = 0
    for (val, conf) in zip(pitch_array, voiced_prob):
        period = sample_rate / val
        frames_.append(PitchFrame(offset, period, conf, 0))
        offset += hop_length
    return frames_


def repair_frames(frames: List[PitchFrame], min_period: float, max_period: float,
                  period_only: bool, outlier_thresh: float = 1.3):
    frames = copy.deepcopy(frames)
    periods = np.array([f.period for f in frames])

    # periods = np.array([max(min_period, min(max_period, f.period)) for f in frames])
    ratio_series = periods[1:] / periods[:-1]
    mask = ratio_series < 1.
    ratio_series[mask] = 1 / ratio_series[mask]
    ratio_outlier = ratio_series > outlier_thresh

    itr = 0
    while itr < len(frames) - 1:
        if ratio_outlier[itr]:
            idx_start = itr

            idx_end = None
            for j in range(itr, len(frames) - 1):
                if not ratio_outlier[j]:
                    idx_end = j
                    break

            cume_offset = frames[itr].offset

            if not idx_end:
                idx_end = idx_start

            try:
                start_period = periods[max(0, itr - 1)]
                end_period = periods[min(len(frames) - 1, idx_end)]

                for i in range(idx_start, idx_end + 1):
                    progress = (i - idx_start) / (idx_end - idx_start + 1)
                    frames[itr].period = (1 - progress) * start_period + progress * end_period
                    if not period_only:
                        frames[itr].offset = round(cume_offset)
                    cume_offset += frames[itr].period
                    itr += 1
            except TypeError or ValueError as v:
                print(v, itr, frames[itr].period, cume_offset)
        itr += 1

    return frames


def filter_atonal_frames(frames: List[PitchFrame],
                         frame_length: int,
                         min_period: float,
                         max_period: float,
                         confidence_filt_thresh=0.05):
    new_frames = copy.deepcopy(frames)
    cume_offset = 0
    i = 0
    while i < len(new_frames):
        frame = new_frames[i]

        if frame.is_atonal(confidence_filt_thresh):
            xy_next = None
            xy_prev = None
            idx_start = i
            idx_end = None

            # seek nearest neighbouring non-outlier frames
            for j in range(i - 1, 0, -1):
                if not frames[j].is_atonal(confidence_filt_thresh):
                    idx_start = j + 1
                    xy_prev = [frames[idx_start].offset, frames[idx_start].period]
                    break

            for j in range(i + 1, len(frames) - 1):
                if not frames[j].is_atonal(confidence_filt_thresh):
                    idx_end = j
                    xy_next = [frames[idx_end].offset, frames[idx_end].period]
                    break

            period = 1.
            if xy_prev and xy_next:
                period = (xy_next[1] + xy_prev[1]) / 2
            elif xy_prev:
                period = xy_prev[1]
                idx_end = len(frames) - 1
            elif xy_next:
                period = xy_next[1]
                idx_start = 0
            else:
                period = 200.
                idx_start = 0

            for k in range(idx_start, idx_end + 1):
                new_frames[i].period = period
                i += 1
        else:
            i += 1

    return new_frames


def fill_frequency_bins(frames: List[PitchFrame], confidence_filt_thresh=0.05) -> List[FrequencyBin]:
    bins = []
    for frame in frames:
        if frame.is_atonal(confidence_filt_thresh):
            continue

        contained = False
        for bin in bins:
            if bin.contains(frame.period):
                bin.accumulate(frame.period)
                contained = True
                break
        if not contained:
            bins.append(FrequencyBin(frame.period))

    return bins


def pitch_detect_crepe(audio: torch.Tensor,
                       sample_rate: int,
                       hop_length: int,
                       fmin: float = 40,
                       fmax: float = 800,
                       ) -> (List, List):
    import torchcrepe

    # Select a model capacity--one of "tiny" or "full"
    model = 'full'

    # Compute pitch and harmonicity
    pitch, harmonicity = torchcrepe.predict(audio.unsqueeze(0), sample_rate, hop_length,
                                            fmin, fmax, model, return_harmonicity=True)

    return pitch.squeeze(0).cpu().numpy(), \
           harmonicity.squeeze(0).cpu().numpy()
