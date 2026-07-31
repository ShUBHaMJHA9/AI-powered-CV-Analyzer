const express = require('express')
const cors = require('cors')
const multer = require('multer')
const axios = require('axios')
const path = require('path')
const fs = require('fs')
require('dotenv').config()

const app = express()
const PORT = process.env.PORT || 5000

// ── Dynamic Auto-Discovery & Health Check for Python AI Service ─
async function checkAiServiceConnection() {
  const candidates = [
    process.env.AI_SERVICE_URL,
    'http://smarrtif_python_ai:8000',
    'http://python-ai:8000',
    'http://127.0.0.1:8081',
    'http://127.0.0.1:8000',
    'http://localhost:8000'
  ].filter(Boolean)

  for (const url of candidates) {
    try {
      const res = await axios.get(`${url}/health`, { timeout: 1500 })
      if (res.status === 200) {
        return { connected: true, url, data: res.data }
      }
    } catch (e) {
      // probe next candidate
    }
  }
  return { connected: false, url: process.env.AI_SERVICE_URL || 'http://smarrtif_python_ai:8000', data: null }
}

async function getAiServiceUrl() {
  const info = await checkAiServiceConnection()
  return info.url
}

// ── Middleware ────────────────────────────────────────────────
app.use(cors())
app.use(express.json())
app.use(express.urlencoded({ extended: true }))

// ── File Upload (Multer) ──────────────────────────────────────
const uploadDir = path.join(__dirname, '../uploads')
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true })

const storage = multer.diskStorage({
  destination: uploadDir,
  filename: (_, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
})

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_, file, cb) => {
    const allowed = ['.pdf', '.docx', '.doc']
    const ext = path.extname(file.originalname).toLowerCase()
    if (allowed.includes(ext)) cb(null, true)
    else cb(new Error('Only PDF and DOCX files are allowed'))
  }
})

// ── Routes ────────────────────────────────────────────────────

// Health check endpoint
app.get(['/health', '/api/health'], async (_, res) => {
  const aiStatus = await checkAiServiceConnection()
  res.json({
    status: 'ok',
    service: 'CV Analyzer Node Backend',
    ai_service: {
      connected: aiStatus.connected,
      url: aiStatus.url,
      status: aiStatus.connected ? 'ready' : 'unreachable'
    },
    author: 'Shubham Kumar Jha',
    credit: 'Made with ❤️ by Shubham Kumar Jha',
    contact_email: 'shubhamjha22088@gmail.com'
  })
})

// Quick parse endpoint
app.post(['/api/parse-cv', '/parse-cv'], upload.single('cv'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'CV file is required' })

    const aiUrl = await getAiServiceUrl()
    const FormData = require('form-data')
    const form = new FormData()
    form.append('cv', fs.createReadStream(req.file.path), { filename: req.file.originalname })

    const { data } = await axios.post(`${aiUrl}/parse-cv`, form, {
      headers: form.getHeaders(),
      timeout: 30000
    })

    fs.unlink(req.file.path, () => { })
    return res.json(data)
  } catch (err) {
    if (req.file) fs.unlink(req.file.path, () => { })
    const msg = err.response?.data?.detail || err.message || 'Quick parse failed'
    return res.status(500).json({ error: msg })
  }
})

// Main analysis endpoint
app.post(['/api/analyze', '/analyze'], upload.single('cv'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'CV file is required' })

    const { target_role = 'data_scientist', github_username, linkedin_data } = req.body
    const aiUrl = await getAiServiceUrl()

    const FormData = require('form-data')
    const form = new FormData()
    form.append('cv', fs.createReadStream(req.file.path), { filename: req.file.originalname })
    form.append('target_role', target_role)
    if (github_username) form.append('github_username', github_username)
    if (linkedin_data) form.append('linkedin_data', linkedin_data)

    const { data } = await axios.post(`${aiUrl}/analyze`, form, {
      headers: form.getHeaders(),
      timeout: 120000 // 2 min timeout
    })

    fs.unlink(req.file.path, () => { })
    return res.json(data)
  } catch (err) {
    console.error('[/api/analyze]', err.message)
    if (req.file) fs.unlink(req.file.path, () => { })
    const msg = err.response?.data?.detail || err.message || 'Analysis failed'
    return res.status(500).json({ error: msg })
  }
})

// GitHub standalone endpoint
app.get('/api/github/:username', async (req, res) => {
  try {
    const aiUrl = await getAiServiceUrl()
    const { data } = await axios.get(`${aiUrl}/github/${req.params.username}`, { timeout: 15000 })
    res.json(data)
  } catch (err) {
    res.status(500).json({ error: err.response?.data?.detail || 'GitHub fetch failed' })
  }
})

// LinkedIn standalone endpoint
app.get('/api/linkedin', async (req, res) => {
  try {
    const { url } = req.query
    const aiUrl = await getAiServiceUrl()
    const { data } = await axios.get(`${aiUrl}/linkedin?url=${encodeURIComponent(url)}`, { timeout: 15000 })
    res.json(data)
  } catch (err) {
    res.status(500).json({ error: err.response?.data?.detail || 'LinkedIn fetch failed' })
  }
})

// ── Error Handler ─────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  console.error(err)
  res.status(500).json({ error: err.message || 'Internal server error' })
})

app.listen(PORT, async () => {
  console.log(`✅ [NODE BACKEND] Running on http://localhost:${PORT}`)
  const aiStatus = await checkAiServiceConnection()
  if (aiStatus.connected) {
    console.log(`🟢 [AI CONNECTED] Successfully connected to Python AI Engine at ${aiStatus.url}`)
  } else {
    console.log(`🔴 [AI WARNING] Unreachable Python AI Engine on launch (${aiStatus.url}). Will auto-probe on requests!`)
  }
})
