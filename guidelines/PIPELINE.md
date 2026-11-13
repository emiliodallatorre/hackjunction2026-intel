# Project pipeline

This pipeline takes the project from an initialized workspace to a tested, autonomous inspection demo. Policy learning and anomaly detection are developed in parallel and brought together for on-device execution.

## 1. Initialize the project

- Set up the project repository, development environment, and team workflow
- Review the challenge requirements, available hardware, and Robotics AI Suite
- Agree on the end-to-end task: pick a brick, inspect it, and sort it into the good or defective bowl
- Define a reliable completion criterion, such as correct sorting across a representative set of bricks and positions

## 2. Prepare and validate the station

- Inventory the Panther Lake laptop, SO-101 follower and leader arms, wrist camera, top-view camera, inspection camera, and sorting bowls
- Connect and validate the robot, cameras, and software before collecting data
- Calibrate the workspace and confirm that the robot can safely reach the pickup area, inspection view, and both bowls
- Mark a varied set of Duplo bricks as good or defective and document the defect types

## 3. Collect and organize data

- Teleoperate the arm to record demonstrations of picking bricks and presenting them to the inspection camera
- Include variation in brick positions and approach paths so the policy can generalize beyond one setup
- Capture representative inspection images of good and defective bricks
- Keep the inspection examples labeled and the demonstrations organized for training and evaluation

## 4. Train the policy and inspection model in parallel

### Manipulation policy

- Upload demonstrations to Physical AI Studio
- Fine-tune a vision-language-action (VLA) policy on cloud GPUs
- Evaluate whether the policy can reliably pick bricks from varied positions and present them to the inspection camera
- Collect additional demonstrations and retrain if performance is insufficient

### Anomaly detection

- Prepare labeled good and defective brick images for Anomalib
- Train an anomaly detection model to identify defects at the inspection point
- Evaluate it on examples not used for training and select a decision threshold for sorting
- Add examples for missed defects or false alarms and retrain as needed

## 5. Integrate the models and robot workflow

- Load the trained policy and anomaly detector into the application
- Orchestrate the complete cycle: locate and pick a brick, present it to the inspection camera, classify it, and place it in the corresponding bowl
- Handle model outputs and robot failures explicitly, including safe recovery when a pick or placement fails
- Consider using the anomaly score to trigger a re-grasp or another inspection

## 6. Optimize and deploy on the laptop

- Convert or optimize the models with OpenVINO and the relevant OpenVINO Physical AI tools
- Run inference on the Panther Lake laptop's CPU, GPU, and NPU and compare speed and accuracy
- Check that optimization, including any quantization, does not reduce sorting reliability
- Confirm that the complete workflow runs on the laptop without depending on cloud inference

## 7. Test the end-to-end cell

- Run repeated autonomous sorting trials with different brick positions and both classes
- Measure successful picks, inspection accuracy, sorting accuracy, recovery behavior, and cycle time
- Test messier layouts and additional defect types after the baseline task is reliable
- Use failure cases to improve demonstrations, training data, thresholds, or robot sequencing, then repeat the relevant evaluations

## 8. Complete and present the task

- Verify the final system can autonomously pick, inspect, and sort bricks from the agreed test set
- Record the model versions, runtime configuration, and evaluation results
- Demonstrate the full inspection loop and summarize its performance, limitations, and any extensions
