from pathlib import Path
from tdw.robot_creator import RobotCreator
from tdw.librarian import RobotLibrarian
import os
from tdw.librarian import ModelLibrarian
from tdw.librarian import ModelRecord
from pathlib import Path
from json import loads
from tdw.librarian import ModelLibrarian, ModelRecord
from functools import partial
from tdw.asset_bundle_creator import AssetBundleCreator


CUSTOM_MODEL_NAMES = ["table", "red_cube", "blue_cube", "golden_cube"]

ROBOT_LIB_PATH = "robots_custom.json"


def create_assetbundle():
    a = AssetBundleCreator()
    a.obj_to_wrl = partial(a.obj_to_wrl, vhacd_resolution=1000000)

    for model_name in CUSTOM_MODEL_NAMES:
        model_path = Path(".") / "models" / (model_name + ".obj")
        asset_bundle_paths, record_path = a.create_asset_bundle(
            model_path=model_path, cleanup=True
        )
    return


def update_custom_record_to_lib(librarian, record_name):
    record_path = Path.home().joinpath(
        f"asset_bundle_creator/Assets/{record_name}.json"
    )
    record = ModelRecord(loads(record_path.read_text()))
    record.volume = 1
    record.wcategory = "object"
    print(f"Updating {record_name}")
    librarian.add_or_update_record(record=record, overwrite=True, write=True)
    return librarian


def create_custom_model_lib():
    # Convert from a relative to absolute path to load the librarian.
    ModelLibrarian.create_library(
        description="Custom model librarian", path="models_custom.json"
    )
    librarian = ModelLibrarian(library=str(Path("models_custom.json").resolve()))
    for model_name in CUSTOM_MODEL_NAMES:
        librarian = update_custom_record_to_lib(
            librarian=librarian, record_name=model_name
        )
    print(f"Updated {librarian.library}.")
    return librarian


def create_robot():
    os.environ[
        "ROS_PACKAGE_PATH"
    ] = "${ROS_PACKAGE_PATH}:/Users/xufeng/robot_creator/Assets/robots:/Users/xufeng/robot_creator/temp_robots/ur5:/Users/xufeng/robot_creator/temp_robots/robotiq"
    r = RobotCreator()
    record = r.create_asset_bundles(
        # urdf_url="https://github.com/utecrobotics/ur5/blob/master/ur5_description/urdf/ur5_robotiq85_gripper.urdf.xacro",
        urdf_url="https://github.com/ROBOTIS-GIT/open_manipulator_p/blob/master/open_manipulator_p_description/urdf/open_manipulator_p_with_gripper_robot.urdf.xacro",
        xacro_args=None,
        # required_repo_urls={
        #     "robotiq_description": "https://github.com/utecrobotics/robotiq",
        # },
        immovable=True,
        up="y",
    )
    lib = update_robot_lib(record)
    return lib


def update_robot_lib(name="ur5"):
    lib = RobotLibrarian(ROBOT_LIB_PATH)
    record = lib.get_record(name)
    if name == "ur5":
        record.targets.update(
            {
                "upper_arm_link": {"target": -90, "type": "revolute"},
                "forearm_link": {"target": 90, "type": "revolute"},
                "wrist_1_link": {"target": 90, "type": "revolute"},
                "wrist_2_link": {"target": 90, "type": "revolute"},
                "wrist_3_link": {"target": 0, "type": "revolute"},
            }
        )
    elif name == "open_manipulator_p_with_gripper":
        record.targets.update(
            {
                "link3": {"target": 15, "type": "revolute"},
                "link5": {"target": 10, "type": "revolute"},
                "gripper_link": {"target": 20, "type": "revolute"},
                "gripper_sub_link": {"target": 20, "type": "revolute"},
            }
        )
    # Add the record to the local library.
    lib.add_or_update_record(record=record, overwrite=True, write=True)
    return lib


# create_custom_model_lib()
# create_robot()
update_robot_lib("open_manipulator_p_with_gripper")
