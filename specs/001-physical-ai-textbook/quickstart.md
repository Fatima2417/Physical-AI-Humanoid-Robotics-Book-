# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

## Prerequisites

Before starting with the textbook, ensure you have the following installed on your system:

### System Requirements
- Ubuntu 22.04 LTS (recommended) or equivalent Linux distribution
- At least 8GB RAM (16GB recommended for simulation)
- At least 50GB free disk space
- Modern CPU with support for virtualization (if using VMs)

### Software Dependencies
1. **ROS 2 Humble Hawksbill**
   ```bash
   # Follow official ROS 2 installation guide for Ubuntu 22.04
   sudo apt update
   sudo apt install software-properties-common
   sudo add-apt-repository universe
   sudo apt update
   sudo apt install ros-humble-desktop
   source /opt/ros/humble/setup.bash
   ```

2. **Docusaurus (for building textbook)**
   ```bash
   # Install Node.js 18+ and npm
   sudo apt install nodejs npm
   npm install -g docusaurus
   ```

3. **Simulation Environments** (Choose one or both)
   - **Gazebo Fortress**:
     ```bash
     sudo apt install ros-humble-gazebo-*
     ```
   - **Unity 2022.3 LTS** (with Unity Robotics Hub)

4. **NVIDIA Isaac Sim** (if available)
   - Download from NVIDIA Developer website
   - Follow installation guide for your platform

## Getting Started with the Textbook

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Set up Your Development Environment
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Install Python dependencies
pip3 install -r requirements.txt  # if available

# Create a workspace for examples
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 3. Build the Textbook
```bash
cd Book
npm install
npm run build
```

### 4. Run the Textbook Locally
```bash
cd Book
npm start
```
The textbook will be available at `http://localhost:3000`

## First Chapter: Introduction to Physical AI

Navigate to the first chapter to begin your journey:
1. Go to `Book/docs/intro/01_introduction-to-physical-ai.md`
2. Follow the instructions for setting up your first ROS 2 workspace
3. Run the example code provided in the chapter
4. Complete the exercises at the end of the chapter

## Code Examples Structure

All code examples are organized by module:
```
Book/
├── docs/
│   ├── code-examples/
│   │   ├── module1-ros2/
│   │   ├── module2-gazebo-unity/
│   │   ├── module3-isaac/
│   │   └── module4-vla/
```

Each module contains:
- Complete working examples
- Step-by-step instructions
- Expected output for verification

## Simulation Setup

### Gazebo Simulation
1. Launch Gazebo:
   ```bash
   ros2 launch gazebo_ros gazebo.launch.py
   ```

### Unity Simulation (if available)
1. Launch Unity Editor
2. Open the provided Unity project
3. Load the robotics scene

## Troubleshooting

### Common Issues

1. **ROS 2 not found**: Make sure to source the ROS 2 setup file in each new terminal:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Docusaurus build fails**: Check that Node.js and npm are properly installed:
   ```bash
   node --version
   npm --version
   ```

3. **Python packages missing**: Install required Python packages:
   ```bash
   pip3 install rclpy geometry_msgs sensor_msgs
   ```

## Next Steps

After completing this quickstart:
1. Proceed to Chapter 1 in the textbook
2. Set up your development environment as described
3. Begin working through the modules in sequence
4. Complete exercises and mini-projects for each chapter