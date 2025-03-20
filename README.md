# GuideSense

> 🏆 **1st Place at Qualcomm x Microsoft x Northeastern University On-Device AI Builders Hackathon**

Introducing **GuideSense** — sensing obstacles, speaking solutions — your personal navigation companion.

## Overview

GuideSense is a **wheelchair navigation assistant** that uses computer vision and voice control to provide real-time guidance and feedback. Our system achieves exceptional performance:

- **YOLOv8n** object detection: **<40ms inference on Snapdragon X Elite CPU**
- **OpenAI Whisper** voice interface: **<10ms inference on NPU** via **Qualcomm AI Engine Direct SDK with ONNX Runtime QNN**

![Watch the Demo](demo.gif)

## ✨ Features

- **Real-Time Object Detection:** Utilizes YOLO to detect objects in the environment and assess potential obstacles
- **Audio Feedback:** Provides concise audio feedback about surroundings, including object type and distance
- **Voice Activation:** Allows users to activate the system with voice commands ("Go" to start)
- **Responsive Feedback:** Interrupts ongoing audio to provide immediate updates about critical obstacles
- **On-Device Processing:** End-to-end processing with zero cloud dependency for privacy and minimal latency
- **Real-Time Depth Estimation:** Calculates precise distances based on YOLO

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.11
- Working webcam and microphone
- Pyenv for managing Python versions (optional but recommended)
- Install [**Qualcomm AI Engine Direct SDK**](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)
- Download [Whisper-Base-En](https://aihub.qualcomm.com/mobile/models/whisper_base_en) ONNX model

### Installation

1. **Clone the Repository:**

```bash
git clone https://github.com/Hackathon-Team-404/GuideSense.git
cd GuideSense
```

2. **Set Up Python Environment:**

```bash
pyenv install 3.11.11
pyenv local 3.11.11
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

5. **Set Up API Keys:**

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
XAI_API_KEY=your_grok_api_key
```

> Note: For basic functionality without LLM features, you can use an empty key: `OPENAI_API_KEY=""`

## 🚀 Usage

1. **Run the Application:**

```bash
python main.py
```

2. **Voice Commands:**

   - Say **"Go"** to activate the system

3. **Quit the Application:**

   - Press **'q'** in the video window to quit

## 📁 Project Structure

```sh
recognition/
├── main.py                 # Main script to run the application
├── situation_analyzer.py   # Analyzes detected objects and provides guidance
├── audio_feedback.py       # Handles audio feedback and text-to-speech
├── voice_control.py        # Manages voice commands for activation and stopping
├── object_detector.py      # Implements YOLO-based object detection
├── requirements.txt        # Lists all Python dependencies
└── .env                    # Contains API keys (excluded from version control)
```

## ⚡ Performance Metrics

| Component | Performance | Hardware |
|-----------|-------------|----------|
| **YOLOv8n Object Detection** | < 40ms inference time | Snapdragon X Elite CPU |
| **OpenAI Whisper** | < 10ms inference | Qualcomm NPU via AI Engine Direct SDK with ONNX Runtime QNN |
| **System** | End-to-end on-device processing | Zero cloud dependency |

---

## 🔮 Future Work

- Integration with distance sensors (ultrasound, IR) for enhanced spatial awareness
- Implementation of SLAM (Simultaneous Localization and Mapping) for improved navigation

## 👥 Team

| Name | LinkedIn |
|------|----------|
| Tianyu Fang | [LinkedIn](https://www.linkedin.com/in/tianyu-fang-tim/) |
| Anson He | [LinkedIn](https://www.linkedin.com/in/ansonhex/) |
| Dingyang Jin | [LinkedIn](https://www.linkedin.com/in/dingyangjin/) |
| Hao Wu | [LinkedIn](https://www.linkedin.com/in/haowuhw/) |
| Harshil Chudasama | [LinkedIn](https://www.linkedin.com/in/harshil-c/) |


# # 