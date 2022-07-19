from manipulatesound.env.manipulation import (
    StateEnv,
    PixelEnv,
    PixelAudioEnv,
    PixelSpectrogramEnv,
    PushOutTask,
)
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_env import StepType, specs
from manipulatesound.env.wrappers import (
    ExtraFrameStackWrapper,
    SpectrogramFrameStackWrapper,
)
import torchaudio
from tqdm import tqdm
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from matplotlib import pyplot as plt
import torch


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


task = PushOutTask()
# env = StateEnv(task=task)
# env = PixelEnv(task=task, screen_width=128, screen_height=128)
# env = PixelAudioEnv(task, port=9888)
env = PixelSpectrogramEnv(nfft=64, channels=2, task=task)
# env = ExtraFrameStackWrapper(env)
env = SpectrogramFrameStackWrapper(env)
env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
env = ExtendedTimeStepWrapper(env)
env.reset()


j = 0
for t in tqdm(range(150000)):
    j += 1
    if j % 50 == 1:
        env.reset()
    # action_normalized = -np.ones((4,))  # in (-1,1)
    action_normalized = np.array([-1, 1, 1, 1])  # in (-1,1)
    action_normalized = action_normalized + np.random.randn() * 0.1
    # action_normalized = (np.random.random((4,)) - 0.5) * 2
    # robot.step(action_normalized)
    # action_normalized[-2] = -1
    time_step = env.step(action_normalized)
    # spectrogram = time_step.observation["spectrogram"]
    # img = spectrogram[0]
    # plt.figure()
    # plt.imshow(img)
    # plt.show()
    if time_step.last():
        env.reset()
    pass

env.close()
