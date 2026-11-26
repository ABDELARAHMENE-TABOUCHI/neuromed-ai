from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import pickle
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import torch.nn as nn
from torchcrf import CRF
import uvicorn
import os
import io

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. DÃ©finition du modÃ¨le BiLSTMâ€‘CRF (identique Ã  l'entraÃ®nement) ----------
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size)

    def forward(self, x, mask):
        embs = self.embedding(x)
        lstm_out, _ = self.lstm(embs)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(self, x, tags, mask):
        emissions = self.forward(x, mask)
        emissions = emissions.transpose(0, 1)
        tags = tags.transpose(0, 1)
        mask = mask.transpose(0, 1)
        return -self.crf(emissions, tags, mask=mask, summed=False).mean()

    def decode(self, x, mask):
        emissions = self.forward(x, mask)
        emissions = emissions.detach()
        mask = mask.detach()
        best_paths = []

        for emission, sequence_mask in zip(emissions, mask):
            seq_length = int(sequence_mask.sum().item())
            if seq_length == 0:
                best_paths.append([])
                continue

            emission = emission[:seq_length]
            score = self.crf.start_transitions.detach() + emission[0]
            history = []

            for timestep in range(1, seq_length):
                next_score = (
                    score.unsqueeze(1)
                    + self.crf.transitions.detach()
                    + emission[timestep].unsqueeze(0)
                )
                best_score, best_previous = next_score.max(dim=0)
                score = best_score
                history.append(best_previous)

            score += self.crf.end_transitions.detach()
            best_last = int(score.argmax().item())
            best_path = [best_last]

            for best_previous in reversed(history):
                best_last = int(best_previous[best_last].item())
                best_path.append(best_last)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths

# ---------- 2. Chargement des trois modÃ¨les ----------
device = torch.device("cpu")

# --- BiLSTM-CRF ---
with open(os.path.join(BASE_DIR, "word2idx.pkl"), "rb") as f:
    word2idx = pickle.load(f)
with open(os.path.join(BASE_DIR, "tag2idx.pkl"), "rb") as f:
    tag2idx = pickle.load(f)
with open(os.path.join(BASE_DIR, "idx2tag.pkl"), "rb") as f:
    idx2tag = pickle.load(f)

bilstm_model = BiLSTM_CRF(vocab_size=len(word2idx), tagset_size=len(tag2idx)).to(device)
bilstm_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "model.pth"), map_location=device))
bilstm_model.eval()

# --- Transformers ONNX INT8 ---
def load_onnx(name):
    folder = os.path.join(BASE_DIR, "onnx_models")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        os.path.join(folder, name + "_int8.onnx"),
        sess_options=options, providers=["CPUExecutionProvider"]
    )
    tokenizer = Tokenizer.from_file(os.path.join(
        folder, name + "_tokenizer", "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.no_padding()
    return session, tokenizer

biolinkbert_model, biolinkbert_tokenizer = load_onnx("biolinkbert")
pubmedbert_model, pubmedbert_tokenizer = load_onnx("pubmedbert")

# Mapping des labels (identique pour tous)
label_list = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
id2label = {i: l for i, l in enumerate(label_list)}

# ---------- 3. Fonctions de prÃ©diction ----------
def predict_bilstm(text):
    tokens = [m.group() for m in re.finditer(r'\w+|[^\w\s]', text)]
    word_ids = [word2idx.get(t.lower(), word2idx["<UNK>"]) for t in tokens]
    mask = [1] * len(word_ids)
    # Padding Ã  256 pour correspondre Ã  l'entraÃ®nement
    if len(word_ids) < 256:
        word_ids += [word2idx["<PAD>"]] * (256 - len(word_ids))
        mask += [0] * (256 - len(mask))
    else:
        word_ids = word_ids[:256]
        mask = mask[:256]
    input_ids = torch.tensor([word_ids]).to(device)
    mask_t = torch.tensor([mask], dtype=torch.bool).to(device)
    with torch.no_grad():
        pred_ids = bilstm_model.decode(input_ids, mask_t)[0]
    
    # Ne garder que les tokens rÃ©els
    real_len = sum(mask[:len(tokens)])
    pred_ids = pred_ids[:real_len]
    return [(token, idx2tag[pred_id]) for token, pred_id in zip(tokens, pred_ids)]

def predict_transformer(text, model, tokenizer):
    encoding = tokenizer.encode(text)
    arrays = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
    }
    feeds = {i.name: arrays[i.name] for i in model.get_inputs()}
    logits = model.run(["logits"], feeds)[0]
    preds = logits.argmax(axis=-1)[0]
    return [(token, id2label[int(pred)])
            for token, pred in zip(encoding.tokens, preds)
            if token not in ["[CLS]", "[SEP]"] and not token.startswith("[UNK]")]

# ---------- 4. Initialisation FastAPI ----------
@app.post("/predict")
async def predict(text: str = Form(...)):
    return {
        "BiLSTM-CRF": predict_bilstm(text),
        "BioLinkBERT": predict_transformer(text, biolinkbert_model, biolinkbert_tokenizer),
        "PubMedBERT": predict_transformer(text, pubmedbert_model, pubmedbert_tokenizer)
    }

# Load OCR Model once lazily
ocr_processor = None
ocr_model = None

@app.post("/ocr")
async def extract_text_from_image(file: UploadFile = File(...)):
    global ocr_processor, ocr_model
    try:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import io
        
        # Load image from the uploaded bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        if ocr_processor is None:
            # We use Microsoft's TrOCR which is designed specifically for handwritten text
            print("Loading TrOCR model for Handwriting Recognition (this might take a moment)...")
            ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
            
        pixel_values = ocr_processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = ocr_model.generate(pixel_values)
        extracted_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return JSONResponse({"text": extracted_text})
        
    except ImportError:
        return JSONResponse(status_code=500, content={"error": "OCR dependencies missing (transformers, pillow). Please run: pip install pillow"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def get_html():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>NeuroMed NER Analysis API Running. You can <a href='/docs'>access the API docs here</a>. Add an index.html file to view the interface.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
