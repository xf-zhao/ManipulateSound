from collections import namedtuple, OrderedDict
from typing import List
import pandas as pd
import cv2
import torchaudio, torchvision, torch
from scipy import signal
from skimage.transform import resize
import time
import sounddevice as sd
from tdw.object_data.rigidbody import Rigidbody
import PIL.Image
from multiprocessing import Process
import io
import dm_env
from tdw.output_data import (
    OutputData,
    Rigidbodies,
    StaticRobot,
    SegmentationColors,
    StaticRigidbodies,
    RobotJointVelocities,
)
from typing import List, Union, Dict
from pathlib import Path
from PIL.Image import Image
from collections import OrderedDict, deque
from typing import List, Tuple, Union
from dm_env import StepType, specs
import random
import math
import time

import ikpy
import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils

from functools import partial
import numpy as np
from tdw.librarian import RobotLibrarian
from tdw.output_data import OutputData, Bounds
from tdw.librarian import ModelLibrarian
from tdw.add_ons.robot import Robot
import os
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.audio_initializer import AudioInitializer
from tdw.add_ons.py_impact import PyImpact
from tdw.physics_audio.audio_material import AudioMaterial
from tdw.physics_audio.object_audio_static import ObjectAudioStatic
from pathlib import Path
from tdw.add_ons.add_on import AddOn
from platform import system
from tdw.object_data.bound import Bound
from tdw.add_ons.py_impact import PyImpact
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.platforms import SYSTEM_TO_UNITY
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils

TABLE_TOP = 0.71
PROJ_PATH = Path(__file__).resolve().parent.parent.parent
MODELS_PATH = PROJ_PATH / "models"

# relative to the robot init position
ACTION_RANGE_VEC = {
    "x": [-0.35, 0.35],
    "y": [0.05, 0.15],
    "z": [0, 0.7],
    "g": [20, 55],
}

SQUARE_CENTER = (
    np.array(ACTION_RANGE_VEC["x"]).mean(),
    TABLE_TOP,
    np.array(ACTION_RANGE_VEC["z"]).mean() - 0.35,  # relative to robot z position
)

SQUARE_LEN = np.array(ACTION_RANGE_VEC["x"])[1] - np.array(ACTION_RANGE_VEC["x"])[0]

# model_names = ["table", "red_cube", "blue_cube", "golden_cube"]
# model_name = "red_cube"


def ensure_vector3(array_or_vector3):
    if isinstance(array_or_vector3, dict):
        return array_or_vector3
    elif isinstance(array_or_vector3, (np.ndarray, list, tuple)):
        return TDWUtils.array_to_vector3(array_or_vector3)
    else:
        raise TypeError


def ensure_in_limits(x, limits):
    if x < limits[0]:
        return limits[0]
    elif x > limits[1]:
        return limits[1]
    return x


def _convert_for_ikpy(x, y, z):
    # convert input (x,y,z) to the format compatible for ik calculation
    return z, -x, y


def _revert_for_ikpy(x, y, z):
    return -y, z, x


def get_model_path(model_name):
    path = Path.home().joinpath(
        "asset_bundle_creator/Assets/NewAssetBundles/"
        + model_name
        + "/"
        + SYSTEM_TO_UNITY[system()]
        + "/"
        + model_name
    )
    url = "file:///" + str(path.resolve())
    # print(url)
    return url


class LogController(Controller):
    def communicate(self, commands):
        print(commands)
        return super().communicate(commands)


class Properties:
    def __init__(self, properties_csv):
        self.physical_properties = pd.read_csv(properties_csv)

    def get_mass(self, name):
        return self.physical_properties[self.physical_properties["name"] == name][
            "mass"
        ].values[0]

    def get_resonance(self, name):
        return self.physical_properties[self.physical_properties["name"] == name][
            "resonance"
        ].values[0]

    def get_bounciness(self, name):
        return self.physical_properties[self.physical_properties["name"] == name][
            "bounciness"
        ].values[0]

    def get_amp(self, name):
        return self.physical_properties[self.physical_properties["name"] == name]["amp"]

    def get_size(self, name):
        return self.physical_properties[self.physical_properties["name"] == name][
            "size"
        ].values[0]

    def get_material(self, name):
        material = self.physical_properties[self.physical_properties["name"] == name][
            "material"
        ].values[0]
        return AudioMaterial.__dict__[material]


class ObjectStatic:
    def __init__(
        self,
        name,
        library=str(MODELS_PATH / "models_ycb.json"),
        properties_csv=str(MODELS_PATH / "YCB_physical_properties.csv"),
        position=TDWUtils.VECTOR3_ZERO,
        rotation=TDWUtils.VECTOR3_ZERO,
        url=None,
        static_friction=0.45,
        dynamic_friction=0.4,
        mass=None,
        bounciness=None,
        amp=None,
        size=None,
        resonance=None,
        material=None,
    ):

        self.object_id = Controller.get_unique_id()
        self.name = name
        self.librarian = ModelLibrarian(library=str(Path(library).resolve()))
        self.library = library
        self.record = self.librarian.get_record(name=name)
        self.properties = Properties(properties_csv=properties_csv)
        self.url = self.record.get_url() if url is None else url
        self.static_friction = static_friction
        self.dynamic_friction = dynamic_friction
        self.mass = mass if mass else self.properties.get_mass(self.name)
        self.bounciness = (
            bounciness if bounciness else self.properties.get_bounciness(self.name)
        )
        self.amp = amp if amp else self.properties.get_amp(self.name)
        self.resonance = (
            resonance if resonance else self.properties.get_resonance(self.name)
        )
        self.material = (
            material if material else self.properties.get_material(self.name)
        )
        self.size = size if size else self.properties.get_size(self.name)
        self.audio_static = ObjectAudioStatic(
            name=name,
            material=self.material,
            mass=self.mass,
            bounciness=self.bounciness,
            amp=self.amp,
            size=self.size,
            resonance=self.resonance,
            object_id=self.object_id,
        )
        self.set_position(position)
        self.set_rotation(rotation)

    @property
    def get_add_commands(self):
        return Controller.get_add_physics_object(
            model_name=self.name,
            object_id=self.object_id,
            position=self.position,
            rotation=self.rotation,
            library=self.library,
            default_physics_values=False,
            mass=self.mass,
            static_friction=self.static_friction,
            dynamic_friction=self.dynamic_friction,
            bounciness=self.bounciness,
        )

    def set_position(self, position):
        self.position = ensure_vector3(position)
        return self

    def set_rotation(self, rotation):
        self.rotation = ensure_vector3(rotation)
        return self


class ArmRobot(Robot):
    def __init__(
        self,
        name,
        robot_id=0,
        position=None,
        rotation=None,
        source=None,
        init_targets=None,
        tip_link=None,
        gripper_link=None,
        chain=None,
        n_action_split=3,
        action_mode="absolute",
    ):
        super().__init__(
            name, robot_id=robot_id, position=position, rotation=rotation, source=source
        )
        if init_targets is not None:
            self._initial_targets = init_targets
        self.targets = {}
        self.tip_link = tip_link
        self.gripper_link = gripper_link
        self.chain = chain
        self._action_mode = action_mode
        self.n_action_split = (
            n_action_split * np.ones((self.action_limits_arr.shape[0],))
            if isinstance(n_action_split, (int, float))
            else n_action_split
        )
        self.action_span = (
            self.action_limits_arr[:, 1] - self.action_limits_arr[:, 0]
        ) / self.n_action_split
        self.movable_links = None
        self.chain_links = None

    def get_initialization_commands(self):
        commands = super().get_initialization_commands()
        commands.append({"$type": "send_robot_joint_velocities", "frequency": "always"})
        return commands

    def action_to_targets(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def action_limits_arr(self):
        limits_arr = []
        for _, limits in ACTION_RANGE_VEC.items():
            limits_arr.append(limits)
        limits_arr = np.array(limits_arr)
        return limits_arr

    def step(self, action):
        # consider just use targets not self.targets
        targets = self.action_to_targets(action)
        self.update_targets(targets)
        self.set_joint_targets(self.targets)

    def update_targets(self, targets):
        self.targets.update(targets)
        return self

    def get_tip_position_relative(self):
        tip_pos = self.get_tip_position()
        return np.array(tip_pos) - TDWUtils.vector3_to_array(self.initial_position)

    def get_tip_position(self):
        x, y, z = self.dynamic.joints[
            self.static.joint_ids_by_name[self.tip_link]
        ].position
        return np.array([x, y, z])

    def get_gripper_open_angle(self):
        g_angle = self.dynamic.joints[
            self.static.joint_ids_by_name[self.gripper_link]
        ].angles[0]
        return g_angle

    def get_gripper_open_angle_normalized(self):
        g_angle = self.get_gripper_open_angle()
        angle_min = ACTION_RANGE_VEC["g"][0]
        angle_max = ACTION_RANGE_VEC["g"][1]
        g_angle_norm = (g_angle - angle_min) / (angle_max - angle_min) * 2 - 1
        return g_angle_norm

    def get_movable_joint_angles(self):
        angles = []
        for joint_name in self.movable_links:
            angle = self.dynamic.joints[
                self.static.joint_ids_by_name[joint_name]
            ].angles
            angles.append(angle)
        return np.array(angles).reshape(-1)

    def get_movable_joint_velocities(self):
        return self._movable_joint_velocities

    def reset(self, position=None, rotation=None):
        self._movable_joint_velocities = np.zeros(shape=(len(self.movable_links),))
        self.targets = {}
        self._set_initial_targets = False
        return super().reset(position=position, rotation=rotation)

    def on_send(self, resp):
        super().on_send(resp)
        joint_velocities = []
        rigidbody_data = {}
        for i in range(len(resp) - 1):
            r_id = OutputData.get_data_type_id(resp[i])
            # Get rigidbody data.
            if r_id == "rojv":
                robot_joint_velocities = RobotJointVelocities(resp[i])
                for j in range(robot_joint_velocities.get_num_joints()):
                    rigidbody_data[robot_joint_velocities.get_joint_id(j)] = Rigidbody(
                        velocity=robot_joint_velocities.get_joint_velocity(j),
                        angular_velocity=robot_joint_velocities.get_joint_angular_velocity(
                            j
                        ),
                        sleeping=robot_joint_velocities.get_joint_sleeping(j),
                    )
        for joint_name in self.movable_links:
            joint_velocities.append(
                rigidbody_data[self.static.joint_ids_by_name[joint_name]].velocity
            )
        self._movable_joint_velocities = np.array(joint_velocities).reshape(-1)
        return


class OMP(ArmRobot):
    def __init__(
        self,
        name="open_manipulator_p_with_gripper",
        tip_link="end_effector_link",
        gripper_link="gripper_link",
        *args,
        **kwargs,
    ):
        chain = ikpy.chain.Chain.from_urdf_file(
            MODELS_PATH / "open_manipulator_p_with_gripper_robot_chain.urdf",
            base_elements=["link1"],
        )
        super().__init__(
            name=name,
            tip_link=tip_link,
            gripper_link=gripper_link,
            chain=chain,
            *args,
            **kwargs,
        )
        self.movable_links = [
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "end_link",
            "gripper_link",
            "gripper_sub_link",
        ]
        self.chain_links = [
            "open_manipulator_p_with_gripper(Clone)",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "end_link",
            "gripper_main_link",
            "end_effector_link",
        ]

    def action_to_targets(self, action):
        # if self._action_mode == "relative":
        #     current_tip_pos = self.get_tip_position()
        #     action = action_normalized * self.action_span
        #     x, y, z = self.get_tip_position_relative()
        #     g = self.get_gripper_open_angle()
        #     action_pose = np.array([x, y, z, g])
        #     new_action_pose = action_pose + action
        # elif self._action_mode == "absolute":
        #     normalized_action_limits = np.array([[-1, 1]] * 4)
        #     k = (self.action_limits_arr[:, 1] - self.action_limits_arr[:, 0]) / (
        #         normalized_action_limits[:, 1] - normalized_action_limits[:, 0]
        #     )
        #     new_action_pose = (
        #         k * (action_normalized - normalized_action_limits[:, 0])
        #         + self.action_limits_arr[:, 0]
        #     )
        # x, y, z, g = [
        #     ensure_in_limits(a, limits)
        #     for a, limits in zip(new_action_pose, self.action_limits_arr)
        # ]
        # print(f"x,y,z,g: {x},{y},{z},{g}")
        x, y, z, g = action
        targets = {}
        ik_angles = self.chain.inverse_kinematics(
            target_position=_convert_for_ikpy(x, y, z),
            # target_orientation=[0, 0, 1],
            # orientation_mode="X",
        )
        ik_angles = np.array([np.rad2deg(angle) for angle in ik_angles])
        for ik_angle, chain_link in zip(ik_angles, self.chain_links):
            joint_id = self.static.joint_ids_by_name[chain_link]
            if self.static.joints[joint_id].joint_type.name != "revolute":
                continue
            targets.update({joint_id: ik_angle})
        targets.update(
            {
                self.static.joint_ids_by_name["gripper_link"]: g,
                self.static.joint_ids_by_name["gripper_sub_link"]: g,
            }
        )
        # print(targets)
        return targets


class UR5(ArmRobot):

    ACTION_RANGE_VEC = {
        "shoulder_link": [-90, 90],
        "upper_arm_link": [-180, 0],
        "forearm_link": [-180, 180],
        "wrist_1_link": [-180, 180],
        "wrist_2_link": [-180, 180],
        "wrist_3_link": [-180, 180],
        "robotiq_85_left_knuckle_link": [0.06, 0.09],
    }

    def __init__(
        self,
        name="ur5",
        tip_link=None,
        gripper_link="robotiq_85_left_knuckle_link",
        *args,
        **kwargs,
    ):

        chain = ikpy.chain.Chain.from_urdf_file(
            MODELS_PATH / "ur5_robotiq85_gripper.urdf", base_elements=["base_link"]
        )
        super().__init__(
            name=name,
            chain=chain,
            tip_link=tip_link,
            gripper_link=gripper_link,
            *args,
            **kwargs,
        )

    def get_gripper_open_length(self):
        g_angle = self.get_gripper_open_angle()
        g_open_length = np.sin(-np.pi * g_angle / 180 + 0.715) * 0.1143 + 0.010
        return g_open_length

    def action_to_targets(self, action_normalized):
        joint_angles = []
        for link in [
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "robotiq_85_left_knuckle_link",
        ]:
            angle = self.dynamic.joints[self.static.joint_ids_by_name[link]].angles[0]
            joint_angles.append(angle)
        action = action_normalized * self.action_span
        action_to = np.array(joint_angles) + action
        for i in range(len(action_to) - 1):
            action_to[i] = ensure_in_limits(action_to[i], self.action_limits_arr[i])
        # special care for gripper
        current_gripper_angle = joint_angles[-1]
        current_gripper_open_length = (
            np.sin(-np.pi * current_gripper_angle / 180 + 0.715) * 0.1143 + 0.010
        )
        open_length = action[-1] + current_gripper_open_length
        open_length = ensure_in_limits(open_length, limits=self.action_limits_arr[-1])
        open_angle = (
            360 * (0.715 - np.arcsin((open_length - 0.010) / 0.1143)) / np.pi
        )  # angle calculation
        action_to[-1] = open_angle
        # TODO: no changes toward current joints angles would create robot movement!! this must be a bug of TDW.
        targets = {
            self.static.joint_ids_by_name["shoulder_link"]: action_to[0],
            self.static.joint_ids_by_name["upper_arm_link"]: action_to[1],
            self.static.joint_ids_by_name["forearm_link"]: action_to[2],
            self.static.joint_ids_by_name["wrist_1_link"]: action_to[3],
            self.static.joint_ids_by_name["wrist_2_link"]: action_to[4],
            self.static.joint_ids_by_name["wrist_3_link"]: action_to[5],
            self.static.joint_ids_by_name["robotiq_85_left_knuckle_link"]: action_to[6],
            self.static.joint_ids_by_name["robotiq_85_right_knuckle_link"]: action_to[
                6
            ],
            self.static.joint_ids_by_name[
                "robotiq_85_left_inner_knuckle_link"
            ]: action_to[6],
            self.static.joint_ids_by_name[
                "robotiq_85_right_inner_knuckle_link"
            ]: action_to[6],
            self.static.joint_ids_by_name[
                "robotiq_85_left_finger_tip_link"
            ]: -action_to[6],
            self.static.joint_ids_by_name[
                "robotiq_85_right_finger_tip_link"
            ]: -action_to[6],
        }
        return targets


class ThirdPersonCameraPlus(ThirdPersonCamera):
    def reset(self):
        pass


class PixelCapture(ImageCapture):
    def __init__(
        self,
        path: Union[str, Path],
        avatar_ids: List[str] = None,
        save=False,
        png: bool = False,
        pass_masks: List[str] = None,
    ):
        super().__init__(
            path=path, avatar_ids=avatar_ids, png=png, pass_masks=pass_masks
        )
        self._save: bool = save
        self.pass_masks = pass_masks

    def get_numpy_images(self):
        images: Dict[str, Dict[str, Image]] = dict()
        for avatar_id in self.images:
            images[avatar_id] = dict()
            for i in range(self.images[avatar_id].get_num_passes()):
                pass_mask = self.images[avatar_id].get_pass_mask(i)
                if pass_mask == "_depth" or pass_mask == "_depth_simple":
                    image = TDWUtils.get_shaped_depth_pass(
                        images=self.images[avatar_id], index=i
                    )
                else:
                    image = np.array(
                        TDWUtils.get_pil_image(images=self.images[avatar_id], index=i)
                    )
                images[avatar_id][self.images[avatar_id].get_pass_mask(i)] = image
        return OrderedDict(images)

    def get_stacked_numpy_images(self):
        stacked_images = []
        images = self.get_numpy_images()
        for avatar_id in images:
            for mask in self.pass_masks:
                stacked_images.append(images[avatar_id][mask])
        stacked_images = np.concatenate(stacked_images, axis=2)
        return stacked_images

    def reset(self):
        pass


class PushOutTask(ObjectManager):
    def __init__(
        self,
        object: ObjectStatic = None,
        ground: ObjectStatic = None,
        transforms: bool = True,
        rigidbodies: bool = True,
        bounds: bool = False,
        object_init_position=(0, TABLE_TOP + 0.05, 0),
        threshold=0.2,
        random_shift=True,
        random_shift_distance_scale=0.1,
    ):
        super().__init__(transforms=transforms, rigidbodies=rigidbodies, bounds=bounds)
        self._done = False
        self.ground = (
            ObjectStatic(
                # y postion found by print(om.transforms[table.object_id].position)
                name="table",
                position=[0, -0.0167, 0],
                mass=1000.0,
                material=AudioMaterial.wood_hard,
                bounciness=0.48,
                amp=0.1,
                size=4,
                resonance=0.3,
            )
            if ground is None
            else ground
        )
        self.object = (
            ObjectStatic(
                name="red_cube",
                position=[0, 1, 0],
                mass=0.1,
                material=AudioMaterial.wood_medium,
                bounciness=0.48,
                amp=0.1,
                size=4,
                resonance=0.3,
            )
            if object is None
            else object
        )
        self._multi = False
        self._shape = 1
        self.threshold = threshold
        self.object_init_position = np.array(object_init_position)
        self._random_shift = random_shift
        self._random_shift_distance_scale = random_shift_distance_scale
        self.reset()

    def get_object_position(self):
        return self.transforms[self.object.object_id].position

    def get_object_velocity(self):
        return np.array(self.rigidbodies[self.object.object_id].velocity)

    def get_initialization_commands(self) -> List[dict]:
        init_commands = [
            {"$type": "destroy_all_objects"},
            {"$type": "remove_position_markers"},
        ]
        init_commands.extend(super().get_initialization_commands())
        if self.ground:
            init_commands.extend(self.ground.get_add_commands)
        if self._multi:
            for obj in self.object:
                init_commands.extend(obj.get_add_commands)
        else:
            init_commands.extend(self.object.get_add_commands)
        # square_y_len = np.array(ACTION_RANGE_VEC['y'])[1] - center[0]
        init_commands.extend(
            [
                {
                    "$type": "add_position_marker",
                    "position": {
                        "x": float(self.object_init_position[0]),
                        # to avoid marker blink
                        "y": self.object_init_position[1] - 0.055,
                        "z": float(self.object_init_position[2]),
                    },
                    "scale": self.threshold * 2,
                    "color": {"r": 1, "g": 0, "b": 0, "a": 0.3},
                    "shape": "circle",
                },
                {
                    "$type": "add_position_marker",
                    "position": {
                        "x": SQUARE_CENTER[0],
                        # to avoid marker blink
                        "y": SQUARE_CENTER[1] - 0.005,
                        "z": SQUARE_CENTER[2],
                    },
                    "scale": SQUARE_LEN,
                    "color": {"r": 0.01, "g": 0.01, "b": 0.01, "a": 0.05},
                    "shape": "square",
                },
            ]
        )
        return init_commands

    def _shift_object_position(self, random_shift_distance_scale, random_shift=True):
        if random_shift:
            shift_angle = random.uniform(0, 2 * np.pi)
            shift_radius = random.uniform(0, random_shift_distance_scale)
            shift_vector = np.array(
                [
                    shift_radius * np.cos(shift_angle),
                    0,
                    shift_radius * np.sin(shift_angle),
                ]
            )
            return shift_vector + self.object_init_position
        return self.object_init_position

    @staticmethod
    def cal_distance(origin, target, ignore_y=True):
        if ignore_y:
            return np.sqrt(
                np.square(origin[0] - target[0]) + np.square(origin[2] - target[2])
            )
        return np.sqrt(np.square(origin - target))

    def get_reward(self):
        reward = -1 / 50
        if self._multi:
            # reward = [reward] * self._shape
            for i, obj in enumerate(self.object):
                distance = self.cal_distance(
                    origin=self.object_init_position,
                    target=self.transforms[obj.object_id].position,
                    ignore_y=True,
                )
                if distance > self.threshold:
                    self._done[i] = True
                    # reward[i] = 1
            if np.sum(self._done) == self._shape:
                reward = 1
            # reward = np.sum(reward) / self._shape
        else:
            distance = self.cal_distance(
                origin=self.object_init_position,
                target=self.transforms[self.object.object_id].position,
                ignore_y=True,
            )
            if distance > self.threshold:
                self._done = True
                reward = 1
        return reward

    def get_discount(self):
        # discount = [1] * self._shape if self._multi else 1
        discount = 1.0
        return discount

    def should_terminate_episode(self):
        if self._multi:
            return np.sum(self._done) == self._shape
        return self._done

    def action_spec(self):
        action_spec = specs.BoundedArray(
            shape=(len(ACTION_RANGE_VEC),),
            dtype=np.float32,
            minimum=[v[0] for _, v in ACTION_RANGE_VEC.items()],
            maximum=[v[1] for _, v in ACTION_RANGE_VEC.items()],
            name="action",
        )
        return action_spec

    def reward_spec(self):
        reward_spec = specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="reward",
        )
        return reward_spec

    def discount_spec(self):
        discount_spec = specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="discount",
        )
        return discount_spec

    def reset(self):
        if isinstance(self.object, List):
            self.object[0].set_position([0.1, TABLE_TOP + 0.001, 0.1])
            self.object[1].set_position([-0.1, TABLE_TOP + 0.001, 0.1])
            self.object[2].set_position([0, TABLE_TOP + 0.001, -0.1])
            self._multi = True
            self._shape = len(self.object)
            self._done = [False] * self._shape
        else:
            shift_object_position = self._shift_object_position(
                self._random_shift_distance_scale, self._random_shift
            )
            self.object.set_position(shift_object_position)
            self._multi = False
            self._done = False
            self._shape = 1
        return super().reset()


class PushOutMultiTask(PushOutTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object = [
            ObjectStatic(
                name="red_cube",
                position=[0, 1, 0],
                mass=0.1,
                material=AudioMaterial.wood_medium,
                bounciness=0.48,
                amp=0.1,
                size=4,
                resonance=0.3,
            ),
            ObjectStatic(
                name="blue_cube",
                position=[0, 1, 0],
                mass=1,
                material=AudioMaterial.ceramic,
                bounciness=0.5,
                amp=0.4,
                size=4,
                resonance=0.7,
            ),
            ObjectStatic(
                name="golden_cube",
                position=[0, 1, 0],
                mass=2,
                material=AudioMaterial.metal,
                bounciness=0.6,
                amp=0.5,
                size=4,
                resonance=0.4,
            ),
        ]
        self.reset()


class PushOutBlueTask(PushOutTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object = ObjectStatic(
            name="blue_cube",
            position=[0, 1, 0],
            mass=1,
            material=AudioMaterial.ceramic,
            bounciness=0.5,
            amp=0.4,
            size=4,
            resonance=0.7,
        )
        self.reset()


class PushOutGoldenTask(PushOutTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object = ObjectStatic(
            name="golden_cube",
            position=[0, 1, 0],
            mass=2,
            material=AudioMaterial.metal,
            bounciness=0.6,
            amp=0.5,
            size=4,
            resonance=0.4,
        )
        self.reset()


class RandomPushOutTask(PushOutTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name_pool = [
            "002_master_chef_can",
            "003_cracker_box",
            "004_sugar_box",
            "005_tomato_soup_can",
            "007_tuna_fish_can",
            "009_gelatin_box",
            "010_potted_meat_can",
            "012_strawberry",
            "013_apple",
            "014_lemon",
            "015_peach",
            "016_pear",
            "017_orange",
            "018_plum",
            "024_bowl",
            "025_mug",
            "061_foam_brick",
            "070-b_colored_wood_blocks",
            "073-c_lego_duplo",
            "073-d_lego_duplo",
            "073-e_lego_duplo",
            "073-f_lego_duplo",
            "073-g_lego_duplo",
            "073-h_lego_duplo",
            "073-i_lego_duplo",
            "073-k_lego_duplo",
            "073-l_lego_duplo",
            "077_rubics_cube",
        ]
        len_pool = len(name_pool)
        rnd_num = np.random.randint(len_pool)
        name = name_pool[rnd_num]
        print(f"Using {name}.")
        self.object = objectstatic(name)
        self.reset()


class Env(dm_env.Environment):
    def __init__(
        self,
        task: PushOutTask,
        port=None,
        robot=None,
        time_limit=50,
        random_state=None,
        step_physics=4,
        screen_width=84,
        screen_height=84,
        render_size=256,
    ):
        super().__init__()
        self.screen_width, self.screen_height = screen_width, screen_height
        self.render_size = render_size
        port = np.random.randint(9999, 59999) if (port is None or port < 0) else port
        print(f"TDW using port {port}...")
        self.c = Controller(port=port)
        self.robot = (
            OMP(
                robot_id=Controller.get_unique_id(),
                position={"x": 0, "y": TABLE_TOP, "z": -0.35},
                rotation={"x": 0, "y": 0, "z": 0},
                source=RobotLibrarian(str(MODELS_PATH / "robots_custom.json")),
                action_mode="absolute",
            )
            if robot is None
            else robot
        )
        self._task = task
        self._time_limit = time_limit
        self._cam = ThirdPersonCamera(
            position={"x": 0, "y": 1.5, "z": 0.35},
            look_at=TDWUtils.array_to_vector3([0, TABLE_TOP, 0]),
            # look_at=cube.object_id,
            avatar_id="a",
        )
        self._step_physics = step_physics
        self.initialized = False

    def reset(self):
        if not self.initialized:
            self.c.communicate(
                [
                    TDWUtils.create_empty_room(12, 12),
                    {
                        "$type": "set_screen_size",
                        "width": self.screen_width,
                        "height": self.screen_height,
                    },
                    {"$type": "set_target_framerate", "framerate": 60},
                    {"$type": "set_aperture", "aperture": 100},
                ]
            )
            self.initialized = True
        self._step_counter = 0
        for add_on in self.c.add_ons:
            add_on.initialized = False
        for reset_element in self._reset_elements:
            reset_element.reset()
        self.c.communicate([{"$type": "step_physics", "frames": 25}])
        self.c.communicate([])
        obs = self.get_observation()
        time_step = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=None,
            discount=None,
            observation=obs,
        )
        return time_step

    def step(self, action) -> dm_env.TimeStep:
        self._step_counter += 1
        self.robot.step(action)
        resp = self.c.communicate(
            [{"$type": "step_physics", "frames": self._step_physics},]
        )
        obs = self.get_observation()
        reward = self._task.get_reward()
        discount = self._task.get_discount()
        done = (
            self._task.should_terminate_episode()
            or self._step_counter >= self._time_limit
        )
        step_type = dm_env.StepType.LAST if done else dm_env.StepType.MID
        return dm_env.TimeStep(step_type, reward, discount, obs)

    def get_observation(self):
        raise NotImplementedError

    def observation_spec(self):
        raise NotImplementedError

    def action_spec(self):
        return self._task.action_spec()

    def reward_spec(self):
        return self._task.reward_spec()

    def discount_spec(self):
        return self._task.get_discount_spec()

    def close(self):
        self.c.communicate({"$type": "terminate"})
        return


class StateEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c.add_ons = [self._task, self._cam, self.robot]
        self._reset_elements = [self._task, self.robot]

    def get_observation(self):
        obs = np.concatenate(
            [
                self._task.get_object_position(),
                self._task.get_object_velocity(),
                self.robot.get_tip_position(),
                self.robot.get_movable_joint_angles(),
                self.robot.get_movable_joint_velocities(),
            ],
            axis=0,
            dtype=np.float32,
        )
        return obs

    def observation_spec(self):
        observation_spec = specs.BoundedArray(
            shape=(25,),
            dtype=np.float32,
            minimum=-10.0,
            maximum=10.0,
            name="observation",
        )
        return observation_spec

    def close(self):
        self.c.communicate({"$type": "terminate"})
        return


class PixelEnv(Env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pass_masks = ["_img"]
        self._pixel_capture = PixelCapture(
            path="/tmp/pixel_capture",
            avatar_ids=[self._cam.avatar_id],
            save=False,
            pass_masks=self._pass_masks,
        )
        self.c.add_ons = [
            self._task,
            self._cam,
            self._pixel_capture,
            self.robot,
        ]
        self._reset_elements = [self._task, self.robot]
        self._time_step = None

    def reset(self):
        time_step = super().reset()
        self._time_step = time_step
        return time_step

    def step(self, action):
        time_step = super().step(action)
        self._time_step = time_step
        return time_step

    def render(self):
        _pixels = self._time_step.observation["pixels"]
        pixels = cv2.resize(
            _pixels,
            dsize=(self.render_size, self.render_size),
            interpolation=cv2.INTER_CUBIC,
        )
        # pixels = self._pixel_capture.get_stacked_numpy_images()
        # frame = cv2.resize(
        #     pixels,
        #     dsize=(self.render_size, self.render_size),
        #     interpolation=cv2.INTER_CUBIC,
        # )
        # return frame
        return pixels

    def get_observation(self):
        pixels = self._pixel_capture.get_stacked_numpy_images()
        obs = OrderedDict({"pixels": pixels})
        return obs

    def observation_spec(self):
        pixels_spec = specs.BoundedArray(
            shape=(self.screen_width, self.screen_height, 3 * len(self._pass_masks),),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels",
        )
        spec = OrderedDict({"pixels": pixels_spec})
        return spec

    def close(self):
        self.c.communicate({"$type": "terminate"})
        return


class PixelStateEnv(PixelEnv):
    def get_observation(self):
        states = np.concatenate(
            [
                # self._task.get_object_position(),
                # self._task.get_object_velocity(),
                self.robot.get_tip_position(),
                self.robot.get_movable_joint_angles(),
                self.robot.get_movable_joint_velocities(),
            ],
            axis=0,
            dtype=np.float32,
        )  # shape=(19,)
        pixels = self._pixel_capture.get_stacked_numpy_images()
        obs = OrderedDict({"states": states, "pixels": pixels})
        return obs

    def observation_spec(self):
        observation_spec = OrderedDict(
            {
                "states": specs.BoundedArray(
                    shape=(25,),
                    dtype=np.float32,
                    minimum=-1.0,
                    maximum=1.0,
                    name="states",
                ),
                "pixels": specs.BoundedArray(
                    shape=(
                        self.screen_width,
                        self.screen_height,
                        3 * len(self._pass_masks),
                    ),
                    dtype=np.uint8,
                    minimum=0,
                    maximum=255,
                    name="pixels",
                ),
            }
        )
        return observation_spec

    def close(self):
        self.c.communicate({"$type": "terminate"})
        return


class PixelAudioEnv(PixelEnv):
    def __init__(self, channels=1, fake_audio=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration = 0.14
        # self.duration = 0.2
        self.sd_channels = channels
        self.sd_samplerate = 8000
        self.sd_frames = int(self.duration * self.sd_samplerate)
        self.fake_audio = fake_audio
        if not fake_audio:
            self._audio_initializer = AudioInitializer(avatar_id="a")
            static_audio_data_overrides = {}
            if self._task._multi:
                for obj in self._task.object:
                    static_audio_data_overrides[obj.object_id] = obj.audio_static
            else:
                static_audio_data_overrides[
                    self._task.object.object_id
                ] = self._task.object.audio_static
            self._py_impact = PyImpact(
                initial_amp=0.9,
                static_audio_data_overrides=static_audio_data_overrides,
            )
            self.c.add_ons = [
                self._task,
                self._cam,
                self._pixel_capture,
                self._py_impact,
                self._audio_initializer,
                self.robot,
            ]
            self._reset_elements = [self._task, self._py_impact, self.robot]
        else:
            self.c.add_ons = [
                self._task,
                self._cam,
                self._pixel_capture,
                self.robot,
            ]
            self._reset_elements = [self._task, self.robot]

    def reset(self):
        time_step = super().reset()
        if "audios" not in time_step.observation:
            time_step.observation["audios"] = np.zeros(
                shape=(self.sd_frames, self.sd_channels), dtype=np.float32
            )
        return time_step

    def step(self, action):
        # TODO: optimize audio recording (stream) someday
        # a = time.time()
        audios = np.zeros(shape=(self.sd_frames, self.sd_channels), dtype=np.float32)
        if not self.fake_audio:
            sd.rec(samplerate=self.sd_samplerate, out=audios)
            time_step = super().step(action)
            sd.stop()
        else:
            time_step = super().step(action)
        time_step.observation["audios"] = audios
        # b = time.time()
        # print(b - a)
        return time_step

    def close(self):
        self.c.communicate({"$type": "terminate"})
        return

    def observation_spec(self):
        pixels_spec = specs.BoundedArray(
            shape=(self.screen_width, self.screen_height, 3 * len(self._pass_masks),),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="pixels",
        )
        audios_spec = specs.BoundedArray(
            shape=(self.sd_frames, self.sd_channels),
            dtype=np.float32,
            minimum=-1,
            maximum=1,
            name="audios",
        )
        spec = OrderedDict({"pixels": pixels_spec, "audios": audios_spec})
        spec.name = "observation"
        _shape, _dtype = {}, {}
        for k, v in spec.items():
            _shape[k] = v.shape
            _dtype[k] = v.dtype
        spec.shape = _shape
        spec.dtype = _dtype
        return spec


class PixelSpectrogramEnv(PixelAudioEnv):
    def __init__(self, nfft=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size_freq = int(nfft // 2)
        self._size = (size_freq, size_freq)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=nfft)
        self.resize = torchvision.transforms.Resize(size=self._size)
        self._time_step = None

    def _transform_observation(self, time_step):
        audios = time_step.observation.pop("audios").transpose(1, 0)
        audios_tensor = torch.from_numpy(audios)
        # range in [-1, 1], and -1 means totally silence
        _spect = self.spectrogram(audios_tensor).log2().clip(-200, 0) / 100 + 1
        spect = self.resize(_spect)
        time_step.observation["spectrogram"] = spect.detach().numpy()
        return time_step

    def reset(self):
        _time_step = super().reset()
        time_step = self._transform_observation(_time_step)
        self._time_step = time_step
        return time_step

    def step(self, action):
        _time_step = super().step(action)
        time_step = self._transform_observation(_time_step)
        self._time_step = time_step
        return time_step

    def render(self):
        _pixels = self._time_step.observation["pixels"]
        _spectrogram = (
            (self._time_step.observation["spectrogram"] + 1) * 255 / 2
        )  # 0~255.
        pixels = cv2.resize(
            _pixels,
            dsize=(self.render_size, self.render_size),
            interpolation=cv2.INTER_CUBIC,
        )
        spectrogram = cv2.resize(
            _spectrogram.astype(np.uint8),
            dsize=(self.render_size // 2, self.render_size // 2),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = {"pixels": pixels, "spectrogram": spectrogram}
        return frame

    def observation_spec(self):
        spec = super().observation_spec()
        spectrogram_spec = specs.BoundedArray(
            shape=(self.sd_channels, *self._size),
            dtype=np.float32,
            minimum=-1,
            maximum=1,
            name="spectrogram",
        )
        spec.pop("audios", None)
        spec["spectrogram"] = spectrogram_spec
        spec.name = "observation"
        _shape, _dtype = {}, {}
        for k, v in spec.items():
            _shape[k] = v.shape
            _dtype[k] = v.dtype
        spec.shape = _shape
        spec.dtype = _dtype
        return spec

    def close(self):
        self.c.communicate({"$type": "terminate"})
        return
