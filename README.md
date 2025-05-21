# Análise de Emoções e Atividades em Vídeo

Este projeto realiza a detecção automática de **emoções faciais** e **atividades corporais** em vídeos, utilizando bibliotecas como `DeepFace`, `MediaPipe`, `OpenCV` e `NumPy`. O resultado é salvo como um novo vídeo com sobreposições visuais (emoções) e um **relatório em JSON** com estatísticas detalhadas.

---

## 📌 Funcionalidades

- 🎭 Detecção de emoções como *happy*, *sad*, *angry*, etc.
- 🕺 Reconhecimento de atividades: `moving`, `walking`, `hands_up`, `stopped`, `dancing`, etc.
- ⚠️ Detecção de movimentos anômalos (ex: gestos bruscos).
- 📊 Geração de resumo em JSON com contagem por emoção e atividade.
- 🎥 Exportação de um novo vídeo com rótulos visuais por face appada.

---

## 🧰 Requisitos

- Python ≥ 3.8  
- OpenCV (com suporte a vídeo)  
- MediaPipe  
- DeepFace  
- tqdm  
- NumPy

---

## 🛠️ Instalação

1. Clone este repositório:

```bash
git clone https://github.com/andersoncatao/fiap_tc_f4.git
cd fiap_tc_f4
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## ▶️ Como Usar

1. Coloque seu vídeo na raiz do projeto ou indique o caminho.
2. Execute o script principal:

```bash
python app.py
```

Por padrão, ele analisa o vídeo `input_video.mp4` e gera:

- `output_video.mp4` com caixas e rótulos de emoção
- `summary.json` com estatísticas como:

```json
{
  "video": "input_video.mp4",
  "total_frames": 3326,
  "frames_processed": 3326,
  "anomalies_apped": 12,
  "emotions": {
    "happy": 1003,
    "neutral": 530
  },
  "activities": {
    "walking": 1020,
    "hands_up": 320,
    "dancing": 50
  }
}
```

---

## ⚙️ Parâmetros personalizáveis

Você pode alterar no código:

- `appor_backend='retinaface'` → para `'mediapipe'`, `'opencv'`, etc.
- `app_every_n=10` → quantos quadros pular entre detecções

---

## 📂 Estrutura do Projeto

```
.
├── app.py                 # Script principal
├── output_video.mp4       # (Gerado após execução)
├── summary.json           # (Gerado após execução)
├── requirements.txt
└── README.md
```

---

## ❗ Observações

- DeepFace pode baixar modelos na primeira execução (pasta `.deepface`).
- `cv2.TrackerCSRT_create()` requer OpenCV com contribs — já incluído no `opencv-python` comum.

---

## 🧠 Créditos

- [DeepFace](https://github.com/serengil/deepface) – Análise facial
- [MediaPipe](https://google.github.io/mediapipe/) – Detecção de pose
- [OpenCV](https://opencv.org/) – Processamento de vídeo

---

## 📜 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
