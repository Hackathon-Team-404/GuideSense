# GuideSense

Introducing GuidenSense, sensing obstacles, speaking solutions - your personal navigation companion.
It features **wheelchair navigation assistant** that uses computer vision and voice control to provide real-time guidance and feedback. It leverages **YOLO** for object detection and a **voice interface** for activation and control.

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
   </br>_You can even use an empty key, we will fix this later._

- Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
XAI_API_KEY=your_grok_api_key
```

- Note: Unless you want to use LLM version, you can input an empty key like : OPENAI_API_KEY=""

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

![Watch the Demo](demo.gif)

---

## License

MIT License

Copyright (c) 2025 Team 404

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Feel free to contribute or raise issues to enhance this project!
