from collections import OrderedDict
from typing import Mapping
from dm_control.suite.wrappers import action_scale
import cv2
import dm_env
from collections import OrderedDict, deque
from dm_env import specs

import numpy as np


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtraFrameStackWrapper(FrameStackWrapper):
    def __init__(self, env, num_frames=3, pixels_key="pixels", audios_key="audios"):
        super().__init__(env, num_frames, pixels_key)
        self._audio_frames = deque([], maxlen=num_frames)
        self._audios_key = audios_key
        wrapped_obs_spec = env.observation_spec()
        assert audios_key in wrapped_obs_spec
        pixels_shape = wrapped_obs_spec[pixels_key].shape
        audios_shape = wrapped_obs_spec[audios_key].shape
        # remove batch dim
        if len(audios_shape) == 4:
            audios_shape = audios_shape[1:]
        # remove batch dim
        self._obs_spec = OrderedDict(
            {
                "pixels": specs.BoundedArray(
                    shape=np.concatenate(
                        [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
                    ),
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name="pixels",
                ),
                "audios": specs.BoundedArray(
                    shape=(audios_shape[1] * num_frames, audios_shape[0]),
                    dtype=np.float32,
                    minimum=-1,
                    maximum=1,
                    name="audios",
                ),
            }
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        assert len(self._audio_frames) == self._num_frames
        pixel_obs = np.concatenate(list(self._frames), axis=0)
        audio_obs = np.concatenate(list(self._audio_frames), axis=0)
        time_step.observation["pixels"] = pixel_obs
        time_step.observation["audios"] = audio_obs
        return time_step

    def _extract_audios(self, time_step):
        audios = time_step.observation[self._audios_key]
        # remove batch dim
        if len(audios.shape) == 4:
            audios = audios[0]
        return audios.transpose(1, 0).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        audios = self._extract_audios(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._audio_frames.append(audios)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        audios = self._extract_audios(time_step)
        self._frames.append(pixels)
        self._audio_frames.append(audios)
        return self._transform_observation(time_step)

    def close(self):
        return self._env.close()


class SpectrogramFrameStackWrapper(FrameStackWrapper):
    def __init__(
        self, env, num_frames=3, pixels_key="pixels", spectrogram_key="spectrogram"
    ):
        super().__init__(env, num_frames, pixels_key)
        self._spectrogram_frames = deque([], maxlen=num_frames)
        self._spectrogram_key = spectrogram_key
        wrapped_obs_spec = env.observation_spec()
        assert spectrogram_key in wrapped_obs_spec
        pixels_shape = wrapped_obs_spec[pixels_key].shape
        spectrogram_shape = wrapped_obs_spec[spectrogram_key].shape
        # remove batch dim
        if len(spectrogram_shape) == 4:
            spectrogram_shape = spectrogram_shape[1:]
        # remove batch dim
        self._obs_spec = self._env.observation_spec()
        pixels_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels",
        )
        spectrogram_spec = specs.BoundedArray(
            shape=(spectrogram_shape[0] * num_frames, *spectrogram_shape[1:]),
            dtype=np.float32,
            minimum=-1,
            maximum=1,
            name="spectrogram",
        )
        self._obs_spec["pixels"] = pixels_spec
        self._obs_spec["spectrogram"] = spectrogram_spec
        self._obs_spec.shape = {
            "pixels": pixels_spec.shape,
            "spectrogram": spectrogram_spec.shape,
        }
        self._obs_spec.dtype = {
            "pixels": pixels_spec.dtype,
            "spectrogram": spectrogram_spec.dtype,
        }
        self._time_step = None

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        assert len(self._spectrogram_frames) == self._num_frames
        time_step.observation["pixels"] = np.concatenate(list(self._frames), axis=0)
        time_step.observation["spectrogram"] = np.concatenate(
            list(self._spectrogram_frames), axis=0
        )
        return time_step

    def _extract_spectrogram(self, time_step):
        spectrogram = time_step.observation[self._spectrogram_key]
        # remove batch dim
        if len(spectrogram.shape) == 4:
            spectrogram = spectrogram[0]
        return spectrogram.copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        spectrogram = self._extract_spectrogram(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
            self._spectrogram_frames.append(spectrogram)
        time_step = self._transform_observation(time_step)
        self._time_step = time_step
        return time_step

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        spectrogram = self._extract_spectrogram(time_step)
        self._frames.append(pixels)
        self._spectrogram_frames.append(spectrogram)
        time_step = self._transform_observation(time_step)
        self._time_step = time_step
        return time_step

    def render(self):
        # only show the last frame (new frame)
        _pixels = self._time_step.observation["pixels"][-3:]
        _spectrogram = (self._time_step.observation["spectrogram"][-1] + 1) * 255 / 2
        pixels = cv2.resize(
            _pixels.transpose(1, 2, 0),
            dsize=(self.render_size, self.render_size),
            interpolation=cv2.INTER_CUBIC,
        )
        spectrogram = cv2.resize(
            _spectrogram.astype(np.uint8),  # .transpose(1, 2, 0),
            dsize=(self.render_size // 2, self.render_size // 2),
            interpolation=cv2.INTER_CUBIC,
        )
        spectrogram = np.expand_dims(spectrogram, axis=2)
        frame = {"pixels": pixels, "spectrogram": spectrogram}
        return frame

    def close(self):
        return self._env.close()


class ActionScaleWrapper(action_scale.Wrapper):
    def close(self):
        return self._env.close()


class TVWrapper(dm_env.Environment):
    def __init__(self, env, blank_len=15):
        self._env = env
        self.blank_len = blank_len

    def step(self, action):
        blen = self.blank_len
        time_step = self._env.step(action)
        obs = time_step.observation
        if isinstance(obs, Mapping):
            img = obs["pixels"]
        else:
            img = obs
        # now img size is (84,84,3)
        noise_cube_size = (blen, blen, 3)
        img[:blen, :blen, :] = np.random.randint(0, 255, size=noise_cube_size)
        img[-blen:, -blen:, :] = np.random.randint(0, 255, size=noise_cube_size)
        img[:blen, -blen:, :] = np.random.randint(0, 255, size=noise_cube_size)
        img[-blen:, :blen, :] = np.random.randint(0, 255, size=noise_cube_size)
        return time_step._replace(observation=obs)

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)
