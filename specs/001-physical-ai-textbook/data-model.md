# Data Model: Physical AI & Humanoid Robotics Textbook

## Textbook Module
- **Name**: String (required) - The module title
- **Description**: String (required) - Brief overview of the module content
- **Chapters**: Array of Chapter objects (required) - List of chapters in the module
- **Learning Objectives**: Array of String (required) - Key learning outcomes
- **Prerequisites**: Array of String (optional) - Required prior knowledge
- **Duration**: Integer (optional) - Estimated completion time in hours

## Chapter
- **Title**: String (required) - The chapter title
- **Content**: String (required) - Main content in Markdown format
- **Module**: Textbook Module (required) - Reference to parent module
- **Sequence**: Integer (required) - Order within the module
- **Learning Objectives**: Array of String (required) - Specific learning outcomes
- **Code Samples**: Array of Code Sample objects (optional) - Associated code examples
- **Diagrams**: Array of Diagram objects (optional) - Associated visual aids
- **Exercises**: Array of Exercise objects (optional) - Practice problems
- **Quiz**: Quiz object (optional) - Assessment questions
- **Mini Project**: Mini Project object (optional) - Hands-on project

## Code Sample
- **Title**: String (required) - Brief description of the code
- **Language**: String (required) - Programming language (Python, C++, etc.)
- **Code**: String (required) - The actual code content
- **Description**: String (optional) - Explanation of the code
- **File Path**: String (optional) - Where the code file is located
- **Associated Chapter**: Chapter (required) - Reference to parent chapter

## Diagram
- **Title**: String (required) - Brief description of the diagram
- **Type**: String (required) - Diagram type (ASCII, UML, Architecture, etc.)
- **Content**: String (required) - The diagram representation
- **Description**: String (optional) - Explanation of the diagram
- **Associated Chapter**: Chapter (required) - Reference to parent chapter

## Exercise
- **Title**: String (required) - Brief description of the exercise
- **Description**: String (required) - Detailed exercise instructions
- **Difficulty**: String (required) - Difficulty level (Beginner, Intermediate, Advanced)
- **Solution**: String (optional) - Suggested solution
- **Hints**: Array of String (optional) - Hints for solving the exercise
- **Associated Chapter**: Chapter (required) - Reference to parent chapter

## Quiz
- **Title**: String (required) - Brief description of the quiz
- **Questions**: Array of Quiz Question objects (required) - List of quiz questions
- **Passing Score**: Integer (optional) - Minimum score to pass (default: 80%)
- **Time Limit**: Integer (optional) - Time limit in minutes
- **Associated Chapter**: Chapter (required) - Reference to parent chapter

## Quiz Question
- **Question**: String (required) - The quiz question text
- **Question Type**: String (required) - Type (Multiple Choice, True/False, Short Answer)
- **Options**: Array of String (optional) - For multiple choice questions
- **Correct Answer**: String (required) - The correct answer
- **Explanation**: String (optional) - Explanation of the correct answer
- **Difficulty**: String (required) - Difficulty level (Beginner, Intermediate, Advanced)

## Mini Project
- **Title**: String (required) - Brief description of the project
- **Description**: String (required) - Detailed project requirements
- **Objectives**: Array of String (required) - Learning objectives
- **Requirements**: Array of String (required) - Technical requirements
- **Deliverables**: Array of String (required) - Expected outcomes
- **Timeline**: Integer (optional) - Estimated completion time in hours
- **Evaluation Criteria**: Array of String (optional) - How the project will be evaluated
- **Associated Chapter**: Chapter (required) - Reference to parent chapter

## Robot Model
- **Name**: String (required) - The robot's name
- **URDF File**: String (required) - Path to the URDF description file
- **Description**: String (optional) - Brief description of the robot
- **Joints**: Array of Joint objects (optional) - List of robot joints
- **Links**: Array of Link objects (optional) - List of robot links
- **Sensors**: Array of Sensor objects (optional) - List of robot sensors

## Joint
- **Name**: String (required) - The joint name
- **Type**: String (required) - Joint type (revolute, prismatic, fixed, etc.)
- **Parent Link**: String (required) - Name of the parent link
- **Child Link**: String (required) - Name of the child link
- **Limits**: Joint Limits object (optional) - Joint movement limits

## Joint Limits
- **Lower**: Float (optional) - Lower limit in radians or meters
- **Upper**: Float (optional) - Upper limit in radians or meters
- **Effort**: Float (optional) - Maximum effort
- **Velocity**: Float (optional) - Maximum velocity

## Link
- **Name**: String (required) - The link name
- **Inertial**: Inertial object (optional) - Inertial properties
- **Visual**: Visual object (optional) - Visual representation
- **Collision**: Collision object (optional) - Collision representation

## Sensor
- **Name**: String (required) - The sensor name
- **Type**: String (required) - Sensor type (camera, lidar, imu, etc.)
- **Topic**: String (required) - ROS topic for sensor data
- **Parameters**: Object (optional) - Sensor-specific parameters
- **Associated Robot**: Robot Model (required) - Reference to parent robot

## File Structure Validation Rules
- All Markdown files must follow Docusaurus formatting requirements
- Code samples must be properly fenced with language specification
- File names must be sequentially numbered for proper ordering
- All URDF files must validate using check_urdf tool
- All Python code must pass syntax validation
- All C++ code must pass syntax validation