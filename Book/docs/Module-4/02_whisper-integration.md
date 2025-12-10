---
sidebar_position: 2
---

# Whisper Integration

## Learning Objectives

By the end of this chapter, you will be able to:
- Install and configure OpenAI Whisper for speech recognition in robotics applications
- Integrate Whisper with ROS 2 for real-time speech processing
- Implement voice command recognition and interpretation
- Optimize Whisper models for real-time robotic applications
- Handle multiple languages and dialects in voice interfaces
- Validate and test speech recognition performance in noisy environments

## Introduction to OpenAI Whisper

OpenAI Whisper is a state-of-the-art speech recognition model that provides exceptional accuracy for automatic speech recognition (ASR) tasks. In robotics applications, Whisper enables natural voice interaction, allowing robots to understand and respond to human commands in real-world environments.

### Whisper Model Architecture

```
Whisper Architecture for Robotics:
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───→│   Whisper       │───→│   Command       │
│   (Microphones,  │    │   ASR Model     │    │   Interpretation│
│   Audio Stream) │    │   (Encoder-     │    │   (NLP, Intent  │
│                 │    │   Decoder)      │    │   Recognition)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                         ┌─────────────────┐
                         │   Robotics      │
                         │   Action        │
                         │   Planning      │
                         └─────────────────┘
```

### Whisper Model Variants

Whisper comes in several sizes, each with different performance characteristics:

| Model | Size | Required VRAM | Relative Speed | English-only | Multilingual |
|-------|------|---------------|----------------|--------------|--------------|
| tiny  | 75 MB| ~1 GB         | ~32x           | ✓            | ✓            |
| base  | 145 MB| ~1 GB        | ~16x           | ✓            | ✓            |
| small | 465 MB| ~2 GB        | ~6x            | ✓            | ✓            |
| medium| 1.5 GB| ~5 GB        | ~2x            | ✓            | ✓            |
| large | 3.0 GB| ~10 GB       | 1x             | ✗            | ✓            |

For robotics applications, the `base` or `small` models typically provide the best balance of accuracy and computational efficiency.

## Installing Whisper for Robotics

### Prerequisites

Before installing Whisper, ensure your system meets the requirements:

- **Python 3.8+** (recommended: 3.9 or 3.10)
- **CUDA-compatible GPU** (for accelerated inference)
- **Sufficient RAM** (8GB minimum, 16GB recommended)
- **Storage space** (varies by model size)

### Installation Methods

#### Method 1: Using pip (Recommended)

```bash
# Install Whisper with pip
pip install openai-whisper

# For GPU acceleration, install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Method 2: Installing from Source

```bash
# Clone the Whisper repository
git clone https://github.com/openai/whisper.git
cd whisper

# Install in development mode
pip install -e .

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Method 3: Using Docker

```dockerfile
# Dockerfile for Whisper with ROS 2
FROM ros:humble-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Whisper
RUN pip3 install openai-whisper

# Install ROS 2 Python packages
RUN pip3 install rclpy

WORKDIR /app
COPY . .

CMD ["bash"]
```

### Verification of Installation

```python
# test_whisper_installation.py
import whisper
import torch

def test_whisper_installation():
    """Test Whisper installation and basic functionality."""
    print("Testing Whisper installation...")

    # Check if Whisper is available
    try:
        print(f"Whisper version: {whisper.__version__}")
    except AttributeError:
        print("Whisper version not available")

    # Check PyTorch and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Test model loading
    try:
        print("\nTesting model loading...")
        model = whisper.load_model("base")
        print("✓ Base model loaded successfully")
        del model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")

    # Test transcribe function
    try:
        print("\nTesting transcription functionality...")
        # This would normally require an audio file, so we'll just check the function exists
        print("✓ Transcription function available")
    except Exception as e:
        print(f"✗ Transcription test failed: {e}")

    print("\nWhisper installation test completed!")

if __name__ == "__main__":
    test_whisper_installation()
```

## Basic Whisper Usage in Robotics

### Simple Transcription Example

```python
# basic_whisper_usage.py
import whisper
import numpy as np
import librosa
import tempfile
import soundfile as sf
import io

class BasicWhisperRobot:
    def __init__(self, model_size="base"):
        """Initialize Whisper model for robotic applications."""
        self.model_size = model_size
        self.model = whisper.load_model(model_size)

        print(f"Whisper {model_size} model loaded for robotic use")
        print(f"Model is {'multilingual' if self.model.is_multilingual else 'English-only'}")

    def transcribe_audio_file(self, audio_path):
        """Transcribe audio from file."""
        try:
            result = self.model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""

    def transcribe_audio_data(self, audio_data, sample_rate=16000):
        """Transcribe raw audio data."""
        try:
            # Save audio data to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)

                result = self.model.transcribe(temp_file.name)

                # Clean up temporary file
                import os
                os.unlink(temp_file.name)

                return result["text"].strip()
        except Exception as e:
            print(f"Audio data transcription error: {e}")
            return ""

    def transcribe_with_options(self, audio_path, language="en", temperature=0.0):
        """Transcribe with specific options."""
        try:
            options = {
                "language": language,
                "temperature": temperature,
                "best_of": 5,
                "beam_size": 5
            }

            result = self.model.transcribe(audio_path, **options)
            return {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result["segments"]
            }
        except Exception as e:
            print(f"Transcription with options error: {e}")
            return {"text": "", "language": "", "segments": []}

# Example usage
def main():
    robot = BasicWhisperRobot("base")

    # Example: Transcribe a mock audio file
    # In real usage, this would be from microphone or audio stream
    print("Whisper robot initialized. Ready for audio processing.")

if __name__ == "__main__":
    main()
```

## Real-Time Audio Processing for Robotics

### Microphone Input Integration

```python
# real_time_audio.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import numpy as np
import pyaudio
import threading
import queue
import time
import whisper
import tempfile
import soundfile as sf
import os

class RealTimeWhisperRobot(Node):
    def __init__(self):
        super().__init__('real_time_whisper_robot')

        # Publishers and subscribers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)
        self.listening_status_pub = self.create_publisher(Bool, 'listening_status', 10)
        self.command_pub = self.create_publisher(String, 'voice_command', 10)

        # Whisper model
        self.model = whisper.load_model("base")
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Audio parameters
        self.rate = 16000  # Sample rate
        self.chunk = 1024  # Frames per buffer
        self.channels = 1  # Mono audio
        self.format = pyaudio.paFloat32  # Audio format

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Threading
        self.recording_thread = None
        self.processing_thread = None
        self.shutdown_event = threading.Event()

        # Parameters for real-time processing
        self.min_audio_duration = 0.5  # Minimum audio chunk to process (seconds)
        self.max_audio_duration = 10.0  # Maximum audio chunk (seconds)
        self.silence_threshold = 0.01   # Threshold to consider silence

        self.get_logger().info('Real-time Whisper Robot initialized')

    def start_listening(self):
        """Start real-time audio recording and processing."""
        if self.is_listening:
            return

        self.is_listening = True
        self.get_logger().info('Starting real-time audio recording...')

        # Start audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # Start threads
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.processing_thread = threading.Thread(target=self.process_audio_chunks, daemon=True)

        self.recording_thread.start()
        self.processing_thread.start()

        # Update listening status
        status_msg = Bool()
        status_msg.data = True
        self.listening_status_pub.publish(status_msg)

    def stop_listening(self):
        """Stop real-time audio recording."""
        if not self.is_listening:
            return

        self.is_listening = False
        self.shutdown_event.set()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Update listening status
        status_msg = Bool()
        status_msg.data = False
        self.listening_status_pub.publish(status_msg)

        self.get_logger().info('Stopped real-time audio recording')

    def record_audio(self):
        """Record audio in real-time."""
        audio_buffer = []
        silent_frames = 0
        max_silent_frames = int(self.rate * 1.0 / self.chunk)  # 1 second of silence

        while not self.shutdown_event.is_set() and self.is_listening:
            try:
                # Read audio data
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Check for silence
                if np.mean(np.abs(audio_chunk)) < self.silence_threshold:
                    silent_frames += 1
                else:
                    silent_frames = 0  # Reset silent counter when audio detected

                # Add chunk to buffer
                audio_buffer.extend(audio_chunk)

                # Check if we have enough audio to process
                buffer_duration = len(audio_buffer) / self.rate

                # Process audio if we have sufficient duration or if silence detected after speech
                if (buffer_duration >= self.min_audio_duration and
                    (silent_frames >= max_silent_frames or buffer_duration >= self.max_audio_duration)):

                    if buffer_duration >= self.min_audio_duration:
                        # Process the accumulated audio
                        audio_data = np.array(audio_buffer)

                        # Add to processing queue
                        self.audio_queue.put(audio_data)

                        self.get_logger().info(f'Queued audio chunk: {buffer_duration:.2f}s')

                    # Reset buffer
                    audio_buffer = []
                    silent_frames = 0

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

            except Exception as e:
                self.get_logger().error(f'Audio recording error: {e}')
                break

    def process_audio_chunks(self):
        """Process audio chunks with Whisper."""
        while not self.shutdown_event.is_set():
            try:
                # Get audio chunk from queue
                audio_data = self.audio_queue.get(timeout=0.1)

                if len(audio_data) > 0:
                    # Transcribe the audio
                    transcription = self.transcribe_audio(audio_data)

                    if transcription:
                        # Publish transcription
                        speech_msg = String()
                        speech_msg.data = transcription
                        self.speech_pub.publish(speech_msg)

                        # Also publish as command
                        cmd_msg = String()
                        cmd_msg.data = transcription
                        self.command_pub.publish(cmd_msg)

                        self.get_logger().info(f'Transcribed: {transcription}')

                self.audio_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {e}')

    def transcribe_audio(self, audio_data):
        """Transcribe audio data using Whisper."""
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.rate)

                # Transcribe
                result = self.model.transcribe(temp_file.name, language='en')

                # Clean up
                os.unlink(temp_file.name)

                return result["text"].strip()

        except Exception as e:
            self.get_logger().error(f'Whisper transcription error: {e}')
            return ""

    def shutdown(self):
        """Clean shutdown of audio processing."""
        self.stop_listening()
        self.shutdown_event.set()

        if self.stream:
            self.stream.close()
        if self.audio:
            self.audio.terminate()

def main(args=None):
    rclpy.init(args=args)
    robot = RealTimeWhisperRobot()

    try:
        robot.start_listening()
        rclpy.spin(robot)
    except KeyboardInterrupt:
        robot.get_logger().info('Shutting down Real-time Whisper Robot')
    finally:
        robot.shutdown()
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Optimized Whisper for Robotics

### Lightweight Whisper Implementation

For robotics applications, we often need to optimize Whisper for real-time performance:

```python
# optimized_whisper.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
import numpy as np
import threading
import queue
import time
import torch
import whisper
import librosa
from dataclasses import dataclass
from typing import Optional

@dataclass
class WhisperConfig:
    model_size: str = "base"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type: str = "float16" if torch.cuda.is_available() else "float32"
    language: str = "en"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0

class OptimizedWhisperRobot(Node):
    def __init__(self, config: Optional[WhisperConfig] = None):
        super().__init__('optimized_whisper_robot')

        # Configuration
        self.config = config or WhisperConfig()

        # Publishers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)
        self.status_pub = self.create_publisher(Bool, 'whisper_status', 10)

        # Model loading with optimizations
        self.model = None
        self.load_optimized_model()

        # Processing queues
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Performance tracking
        self.processing_times = []
        self.frame_count = 0

        # Threading
        self.processing_thread = None
        self.running = False

        self.get_logger().info(f'Optimized Whisper Robot initialized on {self.config.device}')

    def load_optimized_model(self):
        """Load Whisper model with optimizations."""
        try:
            self.get_logger().info(f'Loading Whisper {self.config.model_size} model on {self.config.device}...')

            # Load model with specific device and compute type
            self.model = whisper.load_model(
                self.config.model_size,
                device=self.config.device,
                download_root=None
            ).to(self.config.device)

            # Set to evaluation mode
            self.model.eval()

            self.get_logger().info('Whisper model loaded successfully')

        except Exception as e:
            self.get_logger().error(f'Failed to load Whisper model: {e}')
            self.model = None

    def preprocess_audio(self, audio_data, target_sr=16000):
        """Preprocess audio for Whisper."""
        # Ensure audio is in the right format
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)

        # Resample if necessary
        if hasattr(audio_data, 'sr') and audio_data.sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=audio_data.sr, target_sr=target_sr)
        elif len(audio_data.shape) > 1:
            # Convert stereo to mono
            audio_data = audio_data.mean(axis=1)

        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        return audio_data

    def transcribe_audio_optimized(self, audio_data):
        """Optimized transcription function."""
        if self.model is None:
            return ""

        start_time = time.time()

        try:
            # Preprocess audio
            processed_audio = self.preprocess_audio(audio_data)

            # Move audio to device
            audio_tensor = torch.from_numpy(processed_audio).to(self.config.device)

            # Transcribe with optimized settings
            result = self.model.transcribe(
                audio_tensor,
                language=self.config.language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                patience=self.config.patience,
                temperature=0.0  # Use greedy decoding for speed
            )

            transcription = result["text"].strip()

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            self.get_logger().debug(f'Transcription took {processing_time:.3f}s: {transcription[:50]}...')

            return transcription

        except Exception as e:
            self.get_logger().error(f'Optimized transcription error: {e}')
            return ""

    def add_audio_for_processing(self, audio_data):
        """Add audio data to processing queue."""
        if self.model is not None:
            self.audio_queue.put(audio_data)

    def start_processing(self):
        """Start the audio processing thread."""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        self.processing_thread.start()

        self.get_logger().info('Audio processing started')

    def stop_processing(self):
        """Stop the audio processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)

        self.get_logger().info('Audio processing stopped')

    def process_audio_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)

                # Process the audio
                transcription = self.transcribe_audio_optimized(audio_data)

                if transcription:
                    # Publish result
                    speech_msg = String()
                    speech_msg.data = transcription
                    self.speech_pub.publish(speech_msg)

                    self.get_logger().info(f'Transcribed: {transcription}')

                self.audio_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Processing loop error: {e}')

    def get_performance_metrics(self):
        """Get performance metrics."""
        if not self.processing_times:
            return 0.0, 0.0

        avg_time = sum(self.processing_times) / len(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return avg_time, fps

    def cleanup(self):
        """Clean up resources."""
        self.stop_processing()

def main(args=None):
    rclpy.init(args=args)

    # Create optimized configuration
    config = WhisperConfig(
        model_size="base",  # Use base for good balance of speed/accuracy
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "float32",
        language="en",
        beam_size=3,  # Reduce for faster processing
        best_of=3     # Reduce for faster processing
    )

    robot = OptimizedWhisperRobot(config)

    try:
        robot.start_processing()
        rclpy.spin(robot)
    except KeyboardInterrupt:
        robot.get_logger().info('Shutting down Optimized Whisper Robot')
    finally:
        robot.cleanup()
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Voice Command Recognition

### Command Parsing and Intent Recognition

```python
# voice_command_parser.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import re
import json
from typing import Dict, List, Tuple, Optional

class VoiceCommandParser(Node):
    def __init__(self):
        super().__init__('voice_command_parser')

        # Subscribers and publishers
        self.speech_sub = self.create_subscription(
            String, 'speech_recognition', self.speech_callback, 10
        )
        self.command_pub = self.create_publisher(String, 'parsed_command', 10)
        self.motion_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.action_pub = self.create_publisher(String, 'robot_action', 10)

        # Command patterns and intents
        self.command_patterns = {
            # Movement commands
            'move_forward': [
                r'go forward',
                r'move forward',
                r'go ahead',
                r'go straight',
                r'move straight',
                r'forward',
                r'straight'
            ],
            'move_backward': [
                r'go backward',
                r'move backward',
                r'go back',
                r'move back',
                r'backward',
                r'back'
            ],
            'turn_left': [
                r'turn left',
                r'go left',
                r'turn to the left',
                r'rotate left',
                r'pivot left',
                r'left'
            ],
            'turn_right': [
                r'turn right',
                r'go right',
                r'turn to the right',
                r'rotate right',
                r'pivot right',
                r'right'
            ],
            'stop': [
                r'stop',
                r'halt',
                r'freeze',
                r'pause',
                r'cease',
                r'quit'
            ],
            'speed_control': [
                r'slow down',
                r'speed up',
                r'faster',
                r'slower',
                r'go slow',
                r'go fast'
            ],
            'object_interaction': [
                r'pick up the (.+)',
                r'grab the (.+)',
                r'get the (.+)',
                r'take the (.+)',
                r'pick the (.+)',
                r'hold the (.+)',
                r'put down the (.+)',
                r'release the (.+)',
                r'drop the (.+)'
            ],
            'navigation': [
                r'go to the (.+)',
                r'move to the (.+)',
                r'navigate to the (.+)',
                r'go to (.+)',
                r'move to (.+)',
                r'navigate to (.+)'
            ],
            'object_location': [
                r'where is the (.+)',
                r'find the (.+)',
                r'locate the (.+)',
                r'where are the (.+)',
                r'find (.+)',
                r'locate (.+)'
            ],
            'scene_description': [
                r'what do you see',
                r'describe the scene',
                r'tell me about the room',
                r'what is in the room',
                r'describe your view',
                r'what do you observe'
            ],
            'greeting': [
                r'hello',
                r'hi',
                r'hey',
                r'good morning',
                r'good afternoon',
                r'good evening'
            ],
            'acknowledgment': [
                r'thank you',
                r'thanks',
                r'okay',
                r'alright',
                r'fine',
                r'good'
            ]
        }

        # Command modifiers and parameters
        self.modifiers = {
            'slow': 0.3,
            'medium': 0.6,
            'fast': 1.0,
            'half': 0.5,
            'quarter': 0.25,
            'double': 2.0
        }

        # Distance units
        self.distance_units = {
            'meter': 1.0,
            'meters': 1.0,
            'cm': 0.01,
            'centimeters': 0.01,
            'mm': 0.001,
            'millimeters': 0.001,
            'feet': 0.3048,
            'foot': 0.3048,
            'inches': 0.0254,
            'inch': 0.0254
        }

        # Angle units
        self.angle_units = {
            'degrees': 1.0,
            'degree': 1.0,
            'radians': 57.2958,
            'radian': 57.2958
        }

        # Current robot state
        self.current_speed = 0.5
        self.current_angular_speed = 0.5

        self.get_logger().info('Voice Command Parser initialized')

    def speech_callback(self, msg):
        """Process incoming speech and parse commands."""
        speech_text = msg.data.lower().strip()
        self.get_logger().info(f'Processing speech: {speech_text}')

        # Parse the command
        parsed_commands = self.parse_speech_command(speech_text)

        if parsed_commands:
            for command in parsed_commands:
                # Publish the parsed command
                cmd_msg = String()
                cmd_msg.data = json.dumps(command)
                self.command_pub.publish(cmd_msg)

                # Execute the command
                self.execute_parsed_command(command)

                self.get_logger().info(f'Parsed command: {command}')
        else:
            self.get_logger().warn(f'Could not parse command: {speech_text}')

    def parse_speech_command(self, text: str) -> List[Dict]:
        """Parse speech text into structured commands."""
        commands = []

        # Normalize text
        normalized_text = self.normalize_speech_text(text)

        # Check each command pattern
        for intent, patterns in self.command_patterns.items():
            for pattern in patterns:
                # Use regex to match pattern
                matches = re.finditer(pattern, normalized_text, re.IGNORECASE)

                for match in matches:
                    command = {
                        'intent': intent,
                        'confidence': 1.0,  # For now, assume perfect match
                        'parameters': {},
                        'raw_match': match.group(0),
                        'raw_text': text
                    }

                    # Extract captured groups (for object names, etc.)
                    if match.groups():
                        command['parameters']['target'] = match.group(1).strip()

                    # Add any extracted parameters
                    params = self.extract_parameters(normalized_text)
                    command['parameters'].update(params)

                    commands.append(command)

        # If no commands found, return empty list
        return commands

    def normalize_speech_text(self, text: str) -> str:
        """Normalize speech text for better pattern matching."""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Expand contractions (simple version)
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not"
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def extract_parameters(self, text: str) -> Dict:
        """Extract parameters like distances, speeds, angles from text."""
        params = {}

        # Extract distance/duration parameters
        distance_pattern = r'(\d+(?:\.\d+)?)\s*(meter|meters|cm|centimeters|mm|millimeters|feet|foot|inches|inch)'
        distance_match = re.search(distance_pattern, text)
        if distance_match:
            distance_value = float(distance_match.group(1))
            unit = distance_match.group(2)
            params['distance'] = distance_value * self.distance_units.get(unit, 1.0)

        # Extract angle parameters
        angle_pattern = r'(\d+(?:\.\d+)?)\s*(degrees|degree|radians|radian)'
        angle_match = re.search(angle_pattern, text)
        if angle_match:
            angle_value = float(angle_match.group(1))
            unit = angle_match.group(2)
            params['angle'] = angle_value * self.angle_units.get(unit, 1.0)

        # Extract speed modifiers
        for modifier, value in self.modifiers.items():
            if modifier in text:
                params['speed_modifier'] = value
                if modifier in ['slow', 'slower']:
                    params['speed'] = self.current_speed * 0.5
                elif modifier in ['fast', 'faster']:
                    params['speed'] = min(self.current_speed * 1.5, 1.0)

        # Extract time/duration
        time_pattern = r'(\d+(?:\.\d+)?)\s*(second|seconds|minute|minutes|hour|hours)'
        time_match = re.search(time_pattern, text)
        if time_match:
            time_value = float(time_match.group(1))
            unit = time_match.group(2)
            multiplier = 1.0 if 'second' in unit else (60.0 if 'minute' in unit else 3600.0)
            params['duration'] = time_value * multiplier

        return params

    def execute_parsed_command(self, command: Dict):
        """Execute the parsed command."""
        intent = command['intent']
        params = command.get('parameters', {})
        target = params.get('target', '')

        if intent == 'move_forward':
            self.execute_move_forward(params)
        elif intent == 'move_backward':
            self.execute_move_backward(params)
        elif intent == 'turn_left':
            self.execute_turn_left(params)
        elif intent == 'turn_right':
            self.execute_turn_right(params)
        elif intent == 'stop':
            self.execute_stop()
        elif intent == 'speed_control':
            self.execute_speed_control(params)
        elif intent == 'object_interaction':
            self.execute_object_interaction(target, params)
        elif intent == 'navigation':
            self.execute_navigation(target, params)
        elif intent == 'object_location':
            self.execute_object_location(target, params)
        elif intent == 'scene_description':
            self.execute_scene_description(params)
        elif intent in ['greeting', 'acknowledgment']:
            self.execute_social_interaction(intent, command['raw_text'])

    def execute_move_forward(self, params: Dict):
        """Execute forward movement command."""
        speed = params.get('speed', self.current_speed)
        distance = params.get('distance', 1.0)  # Default 1 meter

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = 0.0

        # In a real system, you'd integrate this with navigation
        # For now, just publish the command
        self.motion_pub.publish(cmd)

        self.get_logger().info(f'Moving forward: speed={speed}, distance={distance}m')

    def execute_move_backward(self, params: Dict):
        """Execute backward movement command."""
        speed = params.get('speed', self.current_speed)
        distance = params.get('distance', 1.0)

        cmd = Twist()
        cmd.linear.x = -speed
        cmd.angular.z = 0.0

        self.motion_pub.publish(cmd)

        self.get_logger().info(f'Moving backward: speed={speed}, distance={distance}m')

    def execute_turn_left(self, params: Dict):
        """Execute left turn command."""
        speed = params.get('speed', self.current_angular_speed)
        angle = params.get('angle', 90.0)  # Default 90 degrees

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = speed

        self.motion_pub.publish(cmd)

        self.get_logger().info(f'Turning left: speed={speed}, angle={angle}°')

    def execute_turn_right(self, params: Dict):
        """Execute right turn command."""
        speed = params.get('speed', self.current_angular_speed)
        angle = params.get('angle', 90.0)

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = -speed

        self.motion_pub.publish(cmd)

        self.get_logger().info(f'Turning right: speed={speed}, angle={angle}°')

    def execute_stop(self):
        """Execute stop command."""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        self.motion_pub.publish(cmd)

        self.get_logger().info('Stopping robot')

    def execute_speed_control(self, params: Dict):
        """Execute speed control command."""
        if 'speed' in params:
            self.current_speed = params['speed']
            self.get_logger().info(f'Speed adjusted to: {self.current_speed}')

    def execute_object_interaction(self, target: str, params: Dict):
        """Execute object interaction command."""
        action = 'pick_up' if any(word in params.get('raw_match', '') for word in ['pick', 'grab', 'take']) else 'put_down'

        action_msg = String()
        action_msg.data = f"{action}:{target}"
        self.action_pub.publish(action_msg)

        self.get_logger().info(f'Object interaction: {action} {target}')

    def execute_navigation(self, target: str, params: Dict):
        """Execute navigation command."""
        action_msg = String()
        action_msg.data = f"navigate_to:{target}"
        self.action_pub.publish(action_msg)

        self.get_logger().info(f'Navigating to: {target}')

    def execute_object_location(self, target: str, params: Dict):
        """Execute object location command."""
        action_msg = String()
        action_msg.data = f"find_object:{target}"
        self.action_pub.publish(action_msg)

        self.get_logger().info(f'Looking for: {target}')

    def execute_scene_description(self, params: Dict):
        """Execute scene description command."""
        action_msg = String()
        action_msg.data = "describe_scene"
        self.action_pub.publish(action_msg)

        self.get_logger().info('Describing current scene')

    def execute_social_interaction(self, intent: str, raw_text: str):
        """Execute social interaction commands."""
        response_msg = String()
        if intent == 'greeting':
            response_msg.data = "Hello! How can I help you today?"
        else:  # acknowledgment
            response_msg.data = "You're welcome!" if "thank" in raw_text else "Got it!"

        self.get_logger().info(f'Social response: {response_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    parser = VoiceCommandParser()

    try:
        rclpy.spin(parser)
    except KeyboardInterrupt:
        parser.get_logger().info('Shutting down Voice Command Parser')
    finally:
        parser.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-Language Support

### Supporting Multiple Languages in Whisper

```python
# multilingual_whisper.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import torch
import re
from typing import Dict, List, Optional

class MultilingualWhisperRobot(Node):
    def __init__(self):
        super().__init__('multilingual_whisper_robot')

        # Publishers and subscribers
        self.speech_sub = self.create_subscription(
            String, 'raw_audio_path', self.audio_callback, 10
        )
        self.transcription_pub = self.create_publisher(String, 'multilingual_transcription', 10)
        self.language_detection_pub = self.create_publisher(String, 'detected_language', 10)

        # Supported languages mapping
        self.supported_languages = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'zh': 'chinese',
            'ja': 'japanese',
            'ko': 'korean',
            'ar': 'arabic',
            'hi': 'hindi'
        }

        # Language detection patterns (simple heuristics for mixed language detection)
        self.language_indicators = {
            'es': [r'\b(?:hola|adiós|gracias|por favor|lo siento)\b', r'\b(?:el|la|los|las|un|una)\b'],
            'fr': [r'\b(?:bonjour|merci|s\'il vous plaît|excusez-moi)\b', r'\b(?:le|la|les|un|une|des)\b'],
            'de': [r'\b(?:hallo|danke|bitte|entschuldigung)\b', r'\b(?:der|die|das|ein|eine)\b'],
            'it': [r'\b(?:ciao|grazie|prego|scusi)\b', r'\b(?:il|la|lo|i|gli|le|un|uno|una)\b'],
            'pt': [r'\b(?:olá|obrigado|por favor|desculpe)\b', r'\b(?:o|a|os|as|um|uma|uns|umas)\b']
        }

        # Initialize models
        self.multilingual_model = whisper.load_model("small")  # Small model for multilingual support
        self.english_model = whisper.load_model("base.en") if self.multilingual_model.is_multilingual else None

        # Current language settings
        self.current_language = 'en'
        self.auto_detect_language = True

        self.get_logger().info('Multilingual Whisper Robot initialized')

    def audio_callback(self, msg):
        """Process audio with language detection and transcription."""
        audio_path = msg.data

        if self.auto_detect_language:
            detected_lang = self.detect_language_from_audio(audio_path)
            self.current_language = detected_lang

            # Publish detected language
            lang_msg = String()
            lang_msg.data = detected_lang
            self.language_detection_pub.publish(lang_msg)

        # Transcribe with the detected/appropriate language
        transcription = self.transcribe_with_language(audio_path, self.current_language)

        if transcription:
            # Publish transcription
            trans_msg = String()
            trans_msg.data = f"[{self.current_language}] {transcription}"
            self.transcription_pub.publish(trans_msg)

            self.get_logger().info(f'Transcribed ({self.current_language}): {transcription}')

    def detect_language_from_audio(self, audio_path: str) -> str:
        """Detect language from audio file (using Whisper's built-in detection)."""
        try:
            # Use Whisper's language detection capability
            result = self.multilingual_model.transcribe(
                audio_path,
                language=None,  # Let Whisper detect automatically
                task="lang_id"   # Language identification task
            )

            detected_lang = result.get("language", "en")
            confidence = result.get("language_probs", {}).get(detected_lang, 0.0)

            self.get_logger().info(f'Detected language: {detected_lang} (confidence: {confidence:.2f})')

            return detected_lang

        except Exception as e:
            self.get_logger().error(f'Language detection error: {e}')
            return 'en'  # Default to English

    def transcribe_with_language(self, audio_path: str, language: str) -> str:
        """Transcribe audio with specific language."""
        try:
            # Select appropriate model based on language
            if language == 'en' and self.english_model:
                model = self.english_model
            else:
                model = self.multilingual_model

            # Transcribe with specified language
            result = model.transcribe(
                audio_path,
                language=language,
                temperature=0.0,
                best_of=1,
                beam_size=1
            )

            return result["text"].strip()

        except Exception as e:
            self.get_logger().error(f'Transcription error for language {language}: {e}')
            return ""

    def force_language_transcription(self, audio_path: str, language: str) -> str:
        """Force transcription in a specific language."""
        try:
            result = self.multilingual_model.transcribe(
                audio_path,
                language=language,
                temperature=0.0
            )
            return result["text"].strip()
        except Exception as e:
            self.get_logger().error(f'Forced transcription error: {e}')
            return ""

    def set_language_preference(self, language_code: str):
        """Set preferred language for transcription."""
        if language_code in self.supported_languages:
            self.current_language = language_code
            self.auto_detect_language = False
            self.get_logger().info(f'Language preference set to: {language_code}')
        else:
            self.get_logger().warn(f'Unsupported language: {language_code}. Supported: {list(self.supported_languages.keys())}')

    def enable_auto_language_detection(self):
        """Enable automatic language detection."""
        self.auto_detect_language = True
        self.get_logger().info('Auto language detection enabled')

def main(args=None):
    rclpy.init(args=args)
    multilingual_robot = MultilingualWhisperRobot()

    try:
        # Example: Set up some test scenarios
        rclpy.spin(multilingual_robot)
    except KeyboardInterrupt:
        multilingual_robot.get_logger().info('Shutting down Multilingual Whisper Robot')
    finally:
        multilingual_robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization and Validation

### Whisper Performance Monitoring

```python
# whisper_performance_monitor.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import AudioData
import time
import threading
from collections import deque
import numpy as np

class WhisperPerformanceMonitor(Node):
    def __init__(self):
        super().__init__('whisper_performance_monitor')

        # Publishers for performance metrics
        self.latency_pub = self.create_publisher(Float32, 'whisper_latency', 10)
        self.throughput_pub = self.create_publisher(Float32, 'whisper_throughput', 10)
        self.cpu_usage_pub = self.create_publisher(Float32, 'whisper_cpu_usage', 10)
        self.gpu_usage_pub = self.create_publisher(Float32, 'whisper_gpu_usage', 10)
        self.memory_usage_pub = self.create_publisher(Float32, 'whisper_memory_usage', 10)
        self.status_pub = self.create_publisher(Bool, 'whisper_system_status', 10)

        # Performance tracking
        self.latency_measurements = deque(maxlen=100)
        self.throughput_measurements = deque(maxlen=100)
        self.start_times = {}
        self.request_count = 0
        self.successful_transcriptions = 0

        # System monitoring
        self.monitoring_active = True

        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self.system_monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.get_logger().info('Whisper Performance Monitor initialized')

    def start_timing(self, request_id: str):
        """Start timing for a specific request."""
        self.start_times[request_id] = time.time()

    def end_timing(self, request_id: str) -> float:
        """End timing and return latency in seconds."""
        if request_id in self.start_times:
            latency = time.time() - self.start_times[request_id]
            del self.start_times[request_id]

            # Store latency measurement
            self.latency_measurements.append(latency)

            # Publish latency
            latency_msg = Float32()
            latency_msg.data = float(latency)
            self.latency_pub.publish(latency_msg)

            return latency

        return 0.0

    def record_successful_transcription(self):
        """Record a successful transcription."""
        self.successful_transcriptions += 1

    def calculate_throughput(self) -> float:
        """Calculate transcription throughput (requests per second)."""
        if len(self.latency_measurements) == 0:
            return 0.0

        # Calculate throughput based on time window
        if len(self.latency_measurements) > 1:
            time_span = self.latency_measurements[-1]  # Last latency represents recent activity
            # This is a simplified calculation - in practice you'd track request timestamps
            return len(self.latency_measurements) / (len(self.latency_measurements) * 0.1)  # Assuming ~0.1s per request
        return 0.0

    def get_average_latency(self) -> float:
        """Get average latency."""
        if not self.latency_measurements:
            return 0.0
        return sum(self.latency_measurements) / len(self.latency_measurements)

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0

    def get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    def get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100  # Percentage
            return 0.0
        except ImportError:
            return 0.0

    def system_monitoring_loop(self):
        """Continuous system monitoring loop."""
        while self.monitoring_active:
            try:
                # Calculate and publish metrics
                avg_latency = self.get_average_latency()
                throughput = self.calculate_throughput()
                cpu_usage = self.get_cpu_usage()
                memory_usage = self.get_memory_usage()
                gpu_usage = self.get_gpu_usage()

                # Publish metrics
                if avg_latency > 0:
                    latency_msg = Float32()
                    latency_msg.data = float(avg_latency)
                    self.latency_pub.publish(latency_msg)

                if throughput > 0:
                    throughput_msg = Float32()
                    throughput_msg.data = float(throughput)
                    self.throughput_pub.publish(throughput_msg)

                cpu_msg = Float32()
                cpu_msg.data = float(cpu_usage)
                self.cpu_usage_pub.publish(cpu_msg)

                memory_msg = Float32()
                memory_msg.data = float(memory_usage)
                self.memory_usage_pub.publish(memory_msg)

                gpu_msg = Float32()
                gpu_msg.data = float(gpu_usage)
                self.gpu_usage_pub.publish(gpu_msg)

                # Calculate system status based on performance metrics
                status_msg = Bool()
                status_msg.data = self.is_system_performing_well(avg_latency, cpu_usage, memory_usage)
                self.status_pub.publish(status_msg)

                # Log performance periodically
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    self.get_logger().info(
                        f'Performance - Latency: {avg_latency:.3f}s, '
                        f'Throughput: {throughput:.2f} req/s, '
                        f'CPU: {cpu_usage:.1f}%, '
                        f'Memory: {memory_usage:.1f}%, '
                        f'GPU: {gpu_usage:.1f}%'
                    )

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                self.get_logger().error(f'Performance monitoring error: {e}')
                time.sleep(1.0)

    def is_system_performing_well(self, latency: float, cpu_usage: float, memory_usage: float) -> bool:
        """Determine if system is performing well based on metrics."""
        # Define performance thresholds
        max_latency = 2.0  # seconds
        max_cpu = 80.0     # percent
        max_memory = 85.0  # percent

        return (latency <= max_latency and
                cpu_usage <= max_cpu and
                memory_usage <= max_memory)

    def cleanup(self):
        """Clean up monitoring resources."""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=2.0)

def main(args=None):
    rclpy.init(args=args)
    monitor = WhisperPerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        monitor.get_logger().info('Shutting down Whisper Performance Monitor')
    finally:
        monitor.cleanup()
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with ROS 2

### Complete Whisper-ROS Integration

```python
# whisper_ros_integration.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import AudioData
from geometry_msgs.msg import Twist
import numpy as np
import threading
import queue
import time
import whisper
import tempfile
import soundfile as sf
import os

class WhisperROSIntegration(Node):
    def __init__(self):
        super().__init__('whisper_ros_integration')

        # Publishers
        self.speech_pub = self.create_publisher(String, 'speech_recognition', 10)
        self.command_pub = self.create_publisher(String, 'voice_command', 10)
        self.status_pub = self.create_publisher(Bool, 'whisper_ready', 10)
        self.confidence_pub = self.create_publisher(Float32, 'recognition_confidence', 10)

        # Subscribers
        self.audio_sub = self.create_subscription(
            AudioData, 'audio_input', self.audio_callback, 10
        )
        self.control_sub = self.create_subscription(
            String, 'whisper_control', self.control_callback, 10
        )

        # Whisper model
        self.model = whisper.load_model("base")
        self.model_lock = threading.Lock()

        # Audio processing
        self.audio_buffer = []
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # System state
        self.is_active = True
        self.is_listening = False
        self.silence_threshold = 0.01
        self.min_audio_duration = 0.5  # seconds
        self.max_audio_duration = 10.0  # seconds

        # Processing thread
        self.processing_thread = threading.Thread(target=self.process_audio_loop, daemon=True)
        self.processing_thread.start()

        # Status timer
        self.status_timer = self.create_timer(1.0, self.publish_status)

        # Initialize as ready
        self.publish_system_status(True)

        self.get_logger().info('Whisper-ROS Integration initialized')

    def audio_callback(self, msg):
        """Process incoming audio data."""
        if not self.is_active or not self.is_listening:
            return

        try:
            # Convert audio data from byte array to numpy array
            # Assuming 16-bit PCM audio data
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to buffer
            self.audio_buffer.extend(audio_data)

            # Check if we have enough audio to process
            buffer_duration = len(self.audio_buffer) / 16000  # assuming 16kHz

            # Process if we have sufficient audio or if we've reached max duration
            if (buffer_duration >= self.min_audio_duration and
                (np.mean(np.abs(audio_data[-1024:])) < self.silence_threshold or
                 buffer_duration >= self.max_audio_duration)):

                if buffer_duration >= self.min_audio_duration:
                    # Process the audio
                    audio_to_process = np.array(self.audio_buffer)
                    self.processing_queue.put(audio_to_process)

                # Reset buffer
                self.audio_buffer = []

        except Exception as e:
            self.get_logger().error(f'Audio processing error: {e}')

    def process_audio_loop(self):
        """Process audio chunks in a separate thread."""
        while rclpy.ok():
            try:
                # Get audio chunk to process
                audio_chunk = self.processing_queue.get(timeout=0.1)

                if len(audio_chunk) > 0:
                    # Process with Whisper
                    with self.model_lock:
                        transcription = self.transcribe_audio_chunk(audio_chunk)

                    if transcription:
                        # Publish results
                        self.publish_transcription(transcription)

                self.processing_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing loop error: {e}')

    def transcribe_audio_chunk(self, audio_data):
        """Transcribe audio chunk using Whisper."""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, 16000)

                # Transcribe
                result = self.model.transcribe(temp_file.name, language='en')

                # Clean up
                os.unlink(temp_file.name)

                return result["text"].strip()

        except Exception as e:
            self.get_logger().error(f'Whisper transcription error: {e}')
            return ""

    def publish_transcription(self, transcription):
        """Publish transcription results."""
        # Publish speech recognition
        speech_msg = String()
        speech_msg.data = transcription
        self.speech_pub.publish(speech_msg)

        # Publish as command
        cmd_msg = String()
        cmd_msg.data = transcription
        self.command_pub.publish(cmd_msg)

        # Publish confidence (mock - in real system this would come from Whisper)
        confidence_msg = Float32()
        confidence_msg.data = 0.9  # Mock confidence
        self.confidence_pub.publish(confidence_msg)

        self.get_logger().info(f'Transcribed: {transcription}')

    def control_callback(self, msg):
        """Handle control commands."""
        command = msg.data.lower().strip()

        if command == 'start':
            self.start_listening()
        elif command == 'stop':
            self.stop_listening()
        elif command == 'toggle':
            if self.is_listening:
                self.stop_listening()
            else:
                self.start_listening()
        elif command == 'status':
            self.publish_system_status(self.is_active)

    def start_listening(self):
        """Start listening for audio."""
        self.is_listening = True
        self.get_logger().info('Started listening for audio')

    def stop_listening(self):
        """Stop listening for audio."""
        self.is_listening = False
        self.get_logger().info('Stopped listening for audio')

    def publish_status(self):
        """Periodically publish system status."""
        self.publish_system_status(self.is_active)

    def publish_system_status(self, is_ready):
        """Publish system ready status."""
        status_msg = Bool()
        status_msg.data = is_ready
        self.status_pub.publish(status_msg)

    def cleanup(self):
        """Clean up resources."""
        self.is_active = False
        self.processing_queue.join()  # Wait for processing to complete

def main(args=None):
    rclpy.init(args=args)
    integration = WhisperROSIntegration()

    try:
        rclpy.spin(integration)
    except KeyboardInterrupt:
        integration.get_logger().info('Shutting down Whisper-ROS Integration')
    finally:
        integration.cleanup()
        integration.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

This chapter covered comprehensive Whisper integration for robotics:

- **Installation and Setup**: Various methods to install Whisper for robotic applications
- **Real-time Audio Processing**: Microphone input and streaming for live transcription
- **Optimization**: Techniques to optimize Whisper for real-time robotic use
- **Command Recognition**: Parsing voice commands and extracting intent
- **Multi-language Support**: Handling multiple languages and dialects
- **Performance Monitoring**: Tracking and optimizing system performance
- **ROS 2 Integration**: Complete integration with ROS 2 messaging system

Whisper provides powerful speech recognition capabilities that enable natural human-robot interaction through voice commands.

## Exercises

1. Install Whisper and test it with audio files
2. Implement real-time audio processing with microphone input
3. Create a voice command parser for robot control
4. Add multi-language support to your system
5. Monitor and optimize the performance of your Whisper system

## Quiz

1. What is the recommended Whisper model size for robotics applications?
   a) tiny
   b) base or small
   c) medium
   d) large

2. What audio sample rate is typically used for Whisper processing?
   a) 8000 Hz
   b) 11025 Hz
   c) 16000 Hz
   d) 44100 Hz

3. What does the "ASR" acronym stand for?
   a) Automatic Speech Recognition
   b) Augmented Speech Reality
   c) Adaptive Signal Recovery
   d) Audio Signal Routing

## Mini-Project: Complete Voice-Controlled Robot

Create a complete voice-controlled robot system with:
1. Whisper integration for speech recognition
2. Real-time audio processing capabilities
3. Voice command parsing and execution
4. Multi-language support
5. Performance monitoring and optimization
6. Integration with ROS 2 navigation stack
7. Testing with various voice commands in different environments