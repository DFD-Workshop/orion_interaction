# orion_interaction

A modular ROS 2 stack for vocal interaction with robots, featuring interchangeable backends
for Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS).
Originally designed for the ORION robot as part of the Tesis-ORION project.

> CURRENTLY UNDER DEVELOPMENT

## Overview

`orion_interaction` replaces the monolithic `orion_chat` package with a clean, decoupled
architecture where each pipeline component exposes an abstract interface.

The pipeline follows this data flow:

```text
Microphone → orion_audio → orion_stt → orion_dialogue → orion_llm
                                              ↓               ↓
                                         orion_tts      orion_actions
                                              ↓               ↓
                                        Audio output   Robot hardware
```

## Requirements

- **ROS 2 Jazzy**
- Python 3.10+
- See each package's README for its specific dependencies

## Packages

| Package | Type | Description |
| --- | --- | --- |
| [`orion_interfaces`](orion_interfaces/README.md) | ament_cmake | Custom messages and services for the pipeline |
| [`orion_audio`](orion_audio/README.md) | ament_python | Microphone capture and Voice Activity Detection (VAD) |
| [`orion_stt`](orion_stt/README.md) | ament_python | Speech-to-Text with interchangeable backends |
| [`orion_llm`](orion_llm/README.md) | ament_python | Large Language Model inference with interchangeable backends |
| [`orion_dialogue`](orion_dialogue/README.md) | ament_python | Conversation manager and intent routing via tool calling |
| [`orion_tts`](orion_tts/README.md) | ament_python | Text-to-Speech with interchangeable backends |
| [`orion_actions`](orion_actions/README.md) | ament_python | Robot action executor and gesture synchronization |
| [`orion_interaction_bringup`](orion_interaction_bringup/README.md) | ament_cmake | Launch files and central configuration |

## Build

```bash
# Clone into your ROS 2 workspace
cd ~/dev_ws/src
git clone <this-repo> orion_interaction

# Build interfaces first, then the rest
colcon build --packages-select orion_interfaces
colcon build --packages-skip orion_interfaces

source ~/dev_ws/install/setup.bash
```

## License

BSD-3-Clause — see [LICENSE](LICENSE).

## Acknowledgements and Derivations

This package builds upon and derives concepts from the following works:

- **[orion_chat](https://github.com/Tesis-ORION/orion_chat)** (Tesis-ORION) — the original
  monolithic interaction stack for the ORION robot. `orion_interaction` is a architectural
  redesign of this work. Licensed under BSD-3-Clause.

- **[audio_messages](https://github.com/Tesis-ORION/audio_messages)** (Tesis-ORION) — defines
  the `AudioInfo` and `AudioData` message types. These definitions have been incorporated
  directly into `orion_interfaces` to eliminate the external dependency. Licensed under
  BSD-3-Clause.

