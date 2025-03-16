# YOLO-and-Audio-Recognition

This project is a **wheelchair navigation assistant** that uses computer vision and voice control to provide real-time guidance and feedback. It leverages **YOLO** for object detection and a **voice interface** for activation and control.

## Features

- **Real-Time Object Detection:** Utilizes YOLO to detect objects in the environment and assess potential obstacles.
- **Audio Feedback:** Provides concise audio feedback about the surroundings, including object type and distance.
- **Voice Activation:** Allows users to activate the system with a voice command and stop it with another command.
- **Responsive Feedback:** Interrupts ongoing audio to provide immediate updates when new conditions are detected.

## Setup Instructions

### Prerequisites

- Python 3.9 or later
- A working webcam and microphone
- Pyenv for managing Python versions (optional but recommended)

### Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/Hackathon-Team-404/YOLO-and-audio-recognition.git
cd YOLO-and-audio-recognition
```

2. **Set Up Python Environment:**

- Use pyenv to install Python 3.9 if not already installed:

```bash
pyenv install 3.9.0
pyenv local 3.9.0
```

3. **Create a Virtual Environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

4. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

5. **Set Up OpenAI API Key:**

- Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. **Run the Application:**

```bash
python main.py
```

2. **Voice Commands:**

- Say **"Go"** to activate the system.
- Say **"I'm here"** to stop the application.

3. **Quit the Application:**

- Press **'q'** in the video window to quit.

## Project Structure

```
recognition/
├── main.py                 # Main script to run the application
├── situation_analyzer.py   # Analyzes detected objects and provides guidance
├── audio_feedback.py       # Handles audio feedback and text-to-speech
├── voice_control.py        # Manages voice commands for activation and stopping
├── object_detector.py      # Implements YOLO-based object detection
├── requirements.txt        # Lists all Python dependencies
└── .env                    # Contains API keys (excluded from version control)
```

---

Feel free to contribute or raise issues to enhance this project!
