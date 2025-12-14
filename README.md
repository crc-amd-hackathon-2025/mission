# AMD Robotics Hackathon 2025 | CRC Assistant

## Team Information

**Team:**
- ID: 1
- Name: CRC
- Members:
    - Cedric Brzyski
    - Remy Sihaan--Gen--Sollen
    - Clement Verrier

**Summary:** Content Creator Assistant

Our work focuses on developing a robotic system that can assist content creators in recording videos: the robot is capable of grabbing a camera, then pointing it towards a specific target (*e.g.*, a person), focusing on it while maintaining a stable shot, and following the target as it moves. Moreover, the robot is controlled through voice commands, allowing for a hands-free experience. This system aims to enhance the content creation process by providing dynamic and adaptive camera work, enabling creators to focus on their performance without worrying about camera operation.

## Submission Details

### 1. Mission Description
Real world application: Content creation assistance (see summary above for details).

### 2. Creativity
- *What is novel or unique in your approach?* We combine voice command recognition with real-time object tracking to create a hands-free camera operation system.
- *Innovation in design, methodology, or application*: A content creator does not need to manually operate the camera, nor does he need to hire a cameraman, as the robot can autonomously handle camera positioning and tracking based on voice commands.

### 3. Technical implementations

- ![Teleoperation / Dataset capture](assets/demo-teleop.gif)
- *Training*: Please refer to [the dedicated Jupyter notebook](https://github.com/crc-amd-hackathon-2025/mission/blob/main/mission/code/training-models-on-rocm.ipynb) (`mission/code/training-models-on-rocm.ipynb`) for the full training pipeline (dataset download, training, artifact upload).
- [Inference](https://drive.google.com/file/d/1LGHCudfnFP8nU68_FdeQEsamhHWYAEHp/view) (playback speed: x2)

### 4. Ease of use
- *How generalizable is your implementation across tasks or environments?*: The system can grab the camera from various positions (as long as it is within reach) and can track different types of targets (people, objects).
- *Flexibility and adaptability of the solution*: The voice command interface allows users to easily switch targets.
- *Types of commands or interfaces needed to control the robot*: Voice commands.

Besides, the YOLO tracking system is compatible with all the COCO labels ("person", "cup", ...).It is compatible with any background qnd requires a minimal calibration. We implemented an automatic calibration tool to make this step easier. The results are stored in a JSON configuration file that can easily be shared and is automatically loaded at restart. 


## Additional Links

- [Video of model picking a camera (playback speed x2)](https://drive.google.com/file/d/1LGHCudfnFP8nU68_FdeQEsamhHWYAEHp/viewLink)
- [Dataset in Hugging Face Hub](https://huggingface.co/datasets/crc-amd-hackathon-2025/grab-cam)
- [Model weights on Hugging Face](https://huggingface.co/crc-amd-hackathon-2025/pi05-grab-cam-2)