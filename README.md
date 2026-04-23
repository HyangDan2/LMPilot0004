# OpenAI Compatible Coding Assistant (PySide6)

A desktop MVP for a local or remote coding assistant using any OpenAI-compatible endpoint.

## Included features
- OOP PySide6 desktop UI
- Base URL / API Key / Model Name fields
- Request input preview shown in the UI
- YAML config save/load
- Test Connection button
- Output stream type support (`stream: true`)
- Current Code / Issue / Import Layer input areas
- Fixed Code output viewer

## Folder structure
```text
openai_compatible_coding_assistant/
├─ app/
│  ├─ core/
│  ├─ services/
│  ├─ ui/
│  ├─ workers/
│  └─ main.py
├─ config/
│  └─ settings.yaml
├─ requirements.txt
└─ README.md
```

## Install
```bash
pip install -r requirements.txt
```

## Run
From the folder **above** `app/`:
```bash
python -m app.main
```

## Default YAML file
The app stores config at:
```text
config/settings.yaml
```

A repository-safe sample config is provided at:
```text
config/settings.example.yaml
```

Session logs are written as CSV files under:
```text
log/
```

## Recommended Base URL examples
### Local llama.cpp server
```text
http://127.0.0.1:8080/v1
```

### NVIDIA/OpenAI-compatible or other gateway
Use the provider's documented OpenAI-compatible `/v1` base URL.

## Notes
- Some providers require `/v1/chat/completions`; this app auto-appends `/chat/completions` if needed.
- The connection test uses `/models` when available.
- Streaming expects Server-Sent Events style `data: {json}` lines.
