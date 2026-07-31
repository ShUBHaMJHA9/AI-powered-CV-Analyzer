import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import './Analyze.css'

const ROLES = [
  'data_scientist', 'ml_engineer', 'frontend_dev', 'backend_dev',
  'full_stack', 'devops', 'data_analyst', 'cloud_architect',
  'product_manager', 'cybersecurity', 'business_analyst', 'ui_ux_designer'
]

const ROLE_LABELS: Record<string, string> = {
  data_scientist: 'Data Scientist', ml_engineer: 'ML Engineer',
  frontend_dev: 'Frontend Developer', backend_dev: 'Backend Developer',
  full_stack: 'Full Stack Developer', devops: 'DevOps Engineer',
  data_analyst: 'Data Analyst', cloud_architect: 'Cloud Architect',
  product_manager: 'Product Manager', cybersecurity: 'Cybersecurity',
  business_analyst: 'Business Analyst', ui_ux_designer: 'UI/UX Designer'
}

type Step = 1 | 2 | 3 | 4

export default function Analyze() {
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [step, setStep] = useState<Step>(1)
  const [file, setFile] = useState<File | null>(null)
  const [role, setRole] = useState('data_scientist')
  const [github, setGithub] = useState('')
  const [linkedinUrl, setLinkedinUrl] = useState('')
  const [linkedinForm, setLinkedinForm] = useState({
    headline: '', skills: '', experience: '', education: '', url: ''
  })
  const [parsingCv, setParsingCv] = useState(false)
  const [autoExtracted, setAutoExtracted] = useState({ github: false, linkedin: false })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [isDragging, setIsDragging] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0]
      validateAndSetFile(selected)
      e.target.value = ''
    }
  }

  const validateAndSetFile = (f: File) => {
    const ext = f.name.substring(f.name.lastIndexOf('.')).toLowerCase()
    if (!['.pdf', '.docx', '.doc'].includes(ext)) {
      setError('Please select a PDF or DOCX file.')
      return
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('File size must be under 10MB.')
      return
    }
    setFile(f)
    setError('')
    quickParseCv(f)
  }

  const getApiUrl = (endpoint: string) => {
    const base = (import.meta.env.VITE_API_URL || 'https://airesume.codetechfoundation.tech').replace(/\/$/, '')
    const cleanEp = endpoint.replace(/^\//, '').replace(/^api\//, '')
    return `${base}/api/${cleanEp}`
  }

  const quickParseCv = async (f: File) => {
    setParsingCv(true)
    try {
      const formData = new FormData()
      formData.append('cv', f)
      const targetUrl = getApiUrl('/api/parse-cv')
      const { data } = await axios.post(targetUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      if (data.github_username) {
        setGithub(data.github_username)
        setAutoExtracted(p => ({ ...p, github: true }))
      }
      if (data.linkedin_url || data.linkedin_username) {
        const url = data.linkedin_url || `https://linkedin.com/in/${data.linkedin_username}`
        setLinkedinUrl(url)
        setLinkedinForm(p => ({
          ...p,
          url,
          headline: data.linkedin_data?.headline || p.headline
        }))
        setAutoExtracted(p => ({ ...p, linkedin: true }))
      }
    } catch (e) {
      console.warn('Quick CV parse warning:', e)
    } finally {
      setParsingCv(false)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0])
    }
  }

  const handleAnalyze = async () => {
    if (!file) { setError('Please upload your CV first.'); return }
    setLoading(true); setError('')
    try {
      const formData = new FormData()
      formData.append('cv', file)
      formData.append('target_role', role)
      if (github) formData.append('github_username', github)

      const finalLinkedinData = {
        ...linkedinForm,
        url: linkedinUrl
      }
      formData.append('linkedin_data', JSON.stringify(finalLinkedinData))

      const targetUrl = getApiUrl('/api/analyze')
      const { data } = await axios.post(targetUrl, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      localStorage.setItem('cv_result', JSON.stringify(data))
      navigate('/dashboard')
    } catch (e: any) {
      setError(e.response?.data?.error || e.message || 'Analysis failed. Make sure backends are running.')
    } finally {
      setLoading(false)
    }
  }

  const STEPS = ['Upload CV', 'Target Role', 'Integrations', 'Analyze']

  return (
    <div className="analyze-page">
      <div className="container">
        <div className="analyze-header fade-up">
          <h1>Analyze Your <span className="text-gradient">Profile</span></h1>
          <p className="text-secondary">AI-powered profile evaluation in seconds</p>
        </div>

        {/* Step Progress */}
        <div className="steps-container fade-up delay-1">
          <div className="steps">
            {STEPS.map((label, i) => (
              <div key={label} className="step-item">
                <div className={`step-circle ${step > i + 1 ? 'done' : step === i + 1 ? 'active' : ''}`}>
                  {step > i + 1 ? '✓' : i + 1}
                </div>
                {i < STEPS.length - 1 && <div className={`step-line ${step > i + 1 ? 'done' : ''}`} />}
              </div>
            ))}
          </div>
          <div className="step-labels">
            {STEPS.map((l, i) => (
              <span key={l} className={`step-label ${step === i + 1 ? 'active' : ''}`}>{l}</span>
            ))}
          </div>
        </div>

        <div className="analyze-card card fade-up delay-2">

          {/* Hidden File Input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,.docx,.doc"
            style={{ display: 'none' }}
          />

          {/* Step 1: Upload */}
          {step === 1 && (
            <div className="step-content">
              <h2 className="step-title">Upload Your CV / Resume</h2>
              <p className="step-desc">Select your resume file to auto-extract GitHub & LinkedIn links</p>

              <div
                className={`dropzone ${isDragging ? 'active' : ''} ${file ? 'has-file' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !file && fileInputRef.current?.click()}
              >
                {file ? (
                  <div className="file-preview-card">
                    <div className="file-icon-box">📄</div>
                    <div className="file-info">
                      <div className="file-name">{file.name}</div>
                      <div className="file-meta">
                        {(file.size / 1024).toFixed(0)} KB · {parsingCv ? '🔍 Scanning GitHub/LinkedIn links...' : '✓ Ready to analyze'}
                      </div>
                    </div>
                    <div className="file-actions">
                      {parsingCv ? (
                        <span className="badge badge-warning"><span className="spinner spinner-sm" /> Scanning</span>
                      ) : (
                        <span className="badge badge-success">✓ Loaded</span>
                      )}
                      <button
                        type="button"
                        className="btn btn-outline btn-sm"
                        onClick={(e) => {
                          e.stopPropagation()
                          setFile(null)
                          setAutoExtracted({ github: false, linkedin: false })
                          if (fileInputRef.current) fileInputRef.current.value = ''
                        }}
                      >
                        Change
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="dropzone-body">
                    <div className="dropzone-icon">📥</div>
                    <h3>Drag & Drop your resume file here</h3>
                    <p className="text-secondary" style={{ marginTop: 6, marginBottom: 16 }}>
                      or click anywhere in this box to choose a file
                    </p>
                    <span className="btn btn-outline btn-sm" style={{ pointerEvents: 'none' }}>
                      📁 Browse File
                    </span>
                    <div className="format-pills" style={{ marginTop: 20 }}>
                      <span className="badge badge-info">PDF</span>
                      <span className="badge badge-info">DOCX</span>
                      <span className="badge badge-info">Up to 10MB</span>
                    </div>
                  </div>
                )}
              </div>

              {error && <div className="error-msg">{error}</div>}

              <div className="step-nav">
                <span />
                <button
                  type="button"
                  className="btn btn-primary btn-lg"
                  disabled={!file || parsingCv}
                  onClick={() => {
                    if (!file) { setError('Please select a CV file to continue.'); return }
                    setError('')
                    setStep(2)
                  }}
                >
                  Next: Target Role →
                </button>
              </div>
            </div>
          )}

          {/* Step 2: Role */}
          {step === 2 && (
            <div className="step-content">
              <h2 className="step-title">Select Target Role</h2>
              <p className="step-desc">Choose the job role you are targeting for tailored evaluation</p>
              <div className="role-grid">
                {ROLES.map(r => (
                  <button
                    key={r}
                    type="button"
                    className={`role-option ${role === r ? 'selected' : ''}`}
                    onClick={() => setRole(r)}
                  >
                    <span>{ROLE_LABELS[r]}</span>
                    {role === r && <span className="role-check">✓</span>}
                  </button>
                ))}
              </div>
              <div className="step-nav">
                <button type="button" className="btn btn-outline" onClick={() => setStep(1)}>← Back</button>
                <button type="button" className="btn btn-primary btn-lg" onClick={() => setStep(3)}>
                  Next: Integrations →
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Integrations */}
          {step === 3 && (
            <div className="step-content">
              <h2 className="step-title">Profile Integrations & Links</h2>
              <p className="step-desc">Auto-extracted from your resume — review, edit, or add details below</p>

              <div className="integration-box">
                <div className="integration-head">
                  <span className="integration-icon">🐙</span>
                  <div style={{ flex: 1 }}>
                    <div className="flex-between">
                      <h3>GitHub Profile</h3>
                      {autoExtracted.github && (
                        <span className="badge badge-success">✨ Extracted from CV</span>
                      )}
                    </div>
                    <p className="text-muted">Adds up to 10 pts based on repos, stars, and code languages</p>
                  </div>
                </div>
                <input
                  type="text"
                  className="form-input"
                  placeholder="GitHub Username (e.g. torvalds)"
                  value={github}
                  onChange={e => {
                    setGithub(e.target.value)
                    setAutoExtracted(p => ({ ...p, github: false }))
                  }}
                />
              </div>

              <div className="integration-box">
                <div className="integration-head">
                  <span className="integration-icon">💼</span>
                  <div style={{ flex: 1 }}>
                    <div className="flex-between">
                      <h3>LinkedIn Profile</h3>
                      {autoExtracted.linkedin && (
                        <span className="badge badge-success">✨ Extracted from CV</span>
                      )}
                    </div>
                    <p className="text-muted">Extracted profile link & intelligence summary</p>
                  </div>
                </div>
                <div className="flex-col gap-3">
                  <input
                    type="text"
                    className="form-input"
                    placeholder="LinkedIn Profile URL (e.g. https://linkedin.com/in/username)"
                    value={linkedinUrl}
                    onChange={e => setLinkedinUrl(e.target.value)}
                  />
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Headline & Job Title (e.g. Senior Developer at Infosys)"
                    value={linkedinForm.headline}
                    onChange={e => setLinkedinForm(p => ({ ...p, headline: e.target.value }))}
                  />
                  <input
                    type="text"
                    className="form-input"
                    placeholder="College / University Degree (e.g. B.Tech Computer Science, IIT Bombay)"
                    value={linkedinForm.education}
                    onChange={e => setLinkedinForm(p => ({ ...p, education: e.target.value }))}
                  />
                  <input
                    type="text"
                    className="form-input"
                    placeholder="Certifications (e.g. AWS Certified Architect, PMP, TensorFlow)"
                    value={linkedinForm.skills}
                    onChange={e => setLinkedinForm(p => ({ ...p, skills: e.target.value }))}
                  />
                </div>
              </div>

              <div className="step-nav">
                <button type="button" className="btn btn-outline" onClick={() => setStep(2)}>← Back</button>
                <button type="button" className="btn btn-primary btn-lg" onClick={() => setStep(4)}>
                  Next: Review & Run →
                </button>
              </div>
            </div>
          )}

          {/* Step 4: Confirm */}
          {step === 4 && (
            <div className="step-content">
              <h2 className="step-title">Ready for AI Evaluation</h2>
              <p className="step-desc">Review your details before running the deep learning pipeline</p>

              <div className="review-grid">
                <div className="review-card">
                  <span className="review-icon">📄</span>
                  <div>
                    <div className="review-label">Selected Resume</div>
                    <div className="review-val">{file?.name}</div>
                  </div>
                </div>
                <div className="review-card">
                  <span className="review-icon">🎯</span>
                  <div>
                    <div className="review-label">Target Role</div>
                    <div className="review-val">{ROLE_LABELS[role]}</div>
                  </div>
                </div>
                <div className="review-card">
                  <span className="review-icon">🐙</span>
                  <div>
                    <div className="review-label">GitHub Account</div>
                    <div className="review-val">{github ? `@${github}` : 'Not provided'}</div>
                  </div>
                </div>
                <div className="review-card">
                  <span className="review-icon">💼</span>
                  <div>
                    <div className="review-label">LinkedIn Profile</div>
                    <div className="review-val">{linkedinUrl || linkedinForm.headline || 'Not provided'}</div>
                  </div>
                </div>
              </div>

              <div className="models-box">
                <h4 style={{ marginBottom: 12, fontSize: '0.92rem', color: 'var(--text-secondary)' }}>
                  Active AI & Deep Learning Models:
                </h4>
                {[
                  { name: 'CVScoringNet', type: 'PyTorch FFNN', pts: '30 pts' },
                  { name: 'SkillMatcherNet', type: 'PyTorch Siamese BiLSTM', pts: '25 pts' },
                  { name: 'ATSClassifierModel', type: 'Keras BiLSTM', pts: '15 pts' },
                  { name: 'ExperienceLevelDetector', type: 'Keras LSTM', pts: '20 pts' },
                  { name: 'GitHub & LinkedIn Analyzer', type: 'Scraper & API', pts: '10 pts' },
                ].map(m => (
                  <div key={m.name} className="model-row">
                    <span className="model-name">{m.name}</span>
                    <span className="badge badge-primary">{m.type}</span>
                    <span className="model-pts">{m.pts}</span>
                  </div>
                ))}
              </div>

              {error && <div className="error-msg">{error}</div>}

              <div className="step-nav">
                <button type="button" className="btn btn-outline" onClick={() => setStep(3)} disabled={loading}>
                  ← Back
                </button>
                <button
                  type="button"
                  className="btn btn-primary btn-lg"
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  {loading ? (
                    <><span className="spinner spinner-sm" /> Running Models...</>
                  ) : (
                    '🚀 Run AI Analysis'
                  )}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
