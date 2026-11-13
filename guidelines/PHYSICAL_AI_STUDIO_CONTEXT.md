# Physical AI Studio context

## Source

- **Article:** "From Zero to Robot Data: Build Your First Imitation Learning Pipeline with Physical AI Studio"
- **Author:** Zhuo Wu
- **Published:** May 7, 2026
- **Article URL:** Not included with the supplied article text
- **Project:** [open-edge-platform/physical-ai-studio](https://github.com/open-edge-platform/physical-ai-studio)

This note summarizes the article's setup workflow for this project; it does not reproduce the article.

## Workflow takeaways

The article focuses on getting from a working application to a first set of usable imitation-learning demonstrations:

1. Start Physical AI Studio before configuring the robot. The article uses the Docker setup, with the XPU profile for Intel GPU acceleration, and serves the application on port `7860`.
2. Create a Studio project for the task before collecting data. Keep the environment, datasets, and models for that task together.
3. Configure and validate the SO-101 follower first: identify the device, assign motor IDs, calibrate, and verify that the UI and physical arm agree. Repeat setup and verification for the leader arm.
4. Add and preview the camera streams needed for the task.
5. Define an environment that binds the follower, leader, and cameras. Its feature set becomes part of the dataset and model input contract.
6. Record demonstration episodes only after the hardware and environment are verified. Reset the scene, check lighting and movement, and accept or discard each take.
7. Vary object positions, orientations, and scene conditions so the policy is not trained only on a single setup.

The article covers application setup through demonstration collection; policy training, anomaly detection, OpenVINO optimization, and autonomous sorting continue in the project pipeline.

## Project-specific notes

- The challenge's Intel Panther Lake laptop should use the XPU service in `infra/docker-compose.yml`.
- The Compose service grants the container privileged access to host devices, following the upstream Studio setup. Run it on the Linux laptop connected to the robot and cameras; Docker Desktop on macOS cannot expose that host's Intel XPU and USB hardware to the container.
- Compose builds the XPU image from the upstream `junction2026` branch, pinned to commit `285924d9381238d3ef2478294c2b85a0eb6ac62b`. Update both build context references together when moving to a newer branch commit.
- The existing `physical-ai-studio-storage` volume and calibration mount are unchanged. Back up the volume before switching versions: the branch may migrate application data, and compatibility with the previous image has not been verified.

From the repository root, build and start the service on the Intel machine with:

```sh
./infra/run_intel_physicalai.sh
```

The image is built from source on startup; the first build may take a while. Subsequent builds can reuse Docker's cache.

The launcher reads the numeric group ID of `/dev/dri/renderD128` on the Linux host. If the Intel GPU uses a different render device, export its numeric group ID as `RENDER_GID` before launching. Docker resolves group names inside the image, so the host's `render` group name may not be available there.

Set `HF_TOKEN` in the root `.env` file when using Hugging Face Hub-backed policies.
