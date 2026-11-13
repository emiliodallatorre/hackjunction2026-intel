# Project pipeline

This pipeline follows the Physical AI Studio setup sequence from application startup through demonstration collection, then continues through policy and anomaly-model training, integration, and autonomous inspection.

## 1. Initialize the project

- Set up the project repository, development environment, and team workflow
- Review the challenge requirements, available hardware, and Robotics AI Suite
- Agree on the end-to-end task: pick a brick, inspect it, and sort it into the good or defective bowl
- Define a reliable completion criterion, such as correct sorting across representative bricks and positions

## 2. Start Physical AI Studio and create a project

- Start Physical AI Studio using the Intel XPU service in `infra/docker-compose.yml`
- Open the application at `http://localhost:7860`
- Create a Studio project for the brick-inspection task before configuring data collection
- Keep the task environment, demonstration datasets, and trained policy associated with this project

## 3. Configure and verify the robot arms

- Connect the SO-101 follower and identify its serial device
- Assign the follower's motor IDs, calibrate its motion range, and verify UI movement against the physical arm
- Save the verified follower, then repeat device identification, motor setup, calibration, and verification for the leader
- Resolve any mismatch before recording demonstrations; calibration errors can degrade all later data

## 4. Add cameras and define the environment

- Add and preview the top-view, wrist, and inspection cameras needed for the task
- Configure an environment that binds the follower, leader, and selected camera streams
- Verify live robot movement and camera feeds together before saving the environment
- Treat the environment's feature set as the input contract for the demonstrations and downstream policy

## 5. Collect and organize data

- Create a dataset using the verified environment
- Record episodes of teleoperating the arm to pick bricks and present them to the inspection camera
- Reset the scene and check lighting and robot movement before each episode; accept useful takes and discard failed ones
- Include variation in brick positions, orientations, and approach paths
- Capture and label representative inspection images of good and defective bricks for the anomaly detector

## 6. Train the policy and inspection model in parallel

### Manipulation policy

- Upload demonstrations to Physical AI Studio
- Fine-tune a vision-language-action (VLA) policy on cloud GPUs
- Evaluate picking and presentation across varied brick positions
- Collect additional demonstrations and retrain if performance is insufficient

### Anomaly detection

- Prepare labeled good and defective brick images for Anomalib
- Train a model to identify defects at the inspection point
- Evaluate it on held-out examples and select a decision threshold for sorting
- Add examples for missed defects or false alarms and retrain as needed

## 7. Integrate the models and robot workflow

- Load the trained policy and anomaly detector into the application
- Orchestrate the complete cycle: locate and pick a brick, present it to the inspection camera, classify it, and place it in the corresponding bowl
- Handle model outputs and robot failures explicitly, including safe recovery when a pick or placement fails
- Consider using the anomaly score to trigger a re-grasp or another inspection

## 8. Optimize and deploy on the laptop

- Convert or optimize the models with OpenVINO and the relevant OpenVINO Physical AI tools
- Run inference on the Panther Lake laptop's CPU, GPU, and NPU and compare speed and accuracy
- Check that optimization, including any quantization, does not reduce sorting reliability
- Confirm that the complete workflow runs on the laptop without depending on cloud inference

## 9. Test the end-to-end cell

- Run repeated autonomous sorting trials with different brick positions and both classes
- Measure successful picks, inspection accuracy, sorting accuracy, recovery behavior, and cycle time
- Test messier layouts and additional defect types after the baseline task is reliable
- Use failure cases to improve demonstrations, training data, thresholds, or robot sequencing, then repeat the relevant evaluations

## 10. Complete and present the task

- Verify the final system can autonomously pick, inspect, and sort bricks from the agreed test set
- Record the model versions, runtime configuration, and evaluation results
- Demonstrate the full inspection loop and summarize its performance, limitations, and any extensions
