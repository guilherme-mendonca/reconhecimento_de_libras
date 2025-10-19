// index.js - servidor Express que recebe uploads e chama script Python para reconhecer
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);

// multer config
const storage = multer.diskStorage({
  destination: function(req, file, cb) {
    cb(null, UPLOAD_DIR);
  },
  filename: function(req, file, cb) {
    // nome único com timestamp
    cb(null, `img_${Date.now()}.jpg`);
  }
});
const upload = multer({ storage: storage });

// memória simples para armazenar letras reconhecidas (ex: última palavra montada)
let letters = []; // ex: ['O','L','A']

// rota para upload (ESP32 faz POST multipart/form-data com campo 'image')
app.post('/upload', upload.single('image'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file' });
  const imgPath = req.file.path;
  console.log('Imagem recebida:', imgPath);

  // chama script Python para reconhecer
  const py = spawn('python', ['recognize.py', imgPath]);

  let out = '', err = '';
  py.stdout.on('data', data => out += data.toString());
  py.stderr.on('data', data => err += data.toString());

  py.on('close', code => {
    if (err) console.error('Python err:', err);
    const letra = out.trim() || null;
    console.log('Letra detectada:', letra);

    if (letra) {
      letters.push(letra);
      // opcional: manter apenas último N
      if (letters.length > 50) letters.shift();
    }

    res.json({ ok: true, letra: letra, all: letters });
    // opcional: remover arquivo para não encher disco
    fs.unlink(imgPath, () => {});
  });
});

// rota para obter letras atuais
app.get('/letters', (req, res) => {
  res.json({ letters });
});

// rota para limpar letras
app.post('/letters/clear', (req, res) => {
  letters = [];
  res.json({ ok: true });
});

// rota para TTS (gera mp3 com gTTS a partir do texto e retorna URL)
app.post('/speak', async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: 'No text' });

  // cria arquivo temporário
  const outFile = path.join(__dirname, 'audio', `tts_${Date.now()}.mp3`);
  if (!fs.existsSync(path.join(__dirname, 'audio'))) fs.mkdirSync(path.join(__dirname, 'audio'));

  // usa script Python gtts_generate.py para gerar o mp3
  const py = spawn('python', ['gtts_generate.py', text, outFile]);
  let out = '', err = '';
  py.stdout.on('data', d => out += d.toString());
  py.stderr.on('data', d => err += d.toString());
  py.on('close', code => {
    if (err) console.error('gTTS err:', err);
    // retorna URL para download
    const url = `/audio/${path.basename(outFile)}`;
    res.json({ ok: true, url });
  });
});

// serve arquivos de áudio gerados
app.use('/audio', express.static(path.join(__dirname, 'audio')));

const PORT = 3000;
app.listen(PORT, () => console.log(`Server rodando em http://0.0.0.0:${PORT}`));
