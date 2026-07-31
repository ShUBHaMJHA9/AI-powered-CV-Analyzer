import { Link } from 'react-router-dom'
import './Landing.css'

const FEATURES = [
  { icon: '🧠', title: 'Deep Learning Powered', desc: 'Custom PyTorch & Keras models trained specifically for CV evaluation — zero external API latency.' },
  { icon: '📊', title: '100-Point Transparent Score', desc: 'Breakdown across CV quality (30pt), skill match (25pt), ATS (15pt), experience (20pt), and GitHub (10pt).' },
  { icon: '🎯', title: 'Targeted Skill Gap Detection', desc: 'spaCy NER matches your extracted skills against 12 industry role templates to list exact missing skills.' },
  { icon: '🤖', title: 'BiLSTM ATS Optimizer', desc: 'Keras Bidirectional LSTM predicts ATS compatibility and highlights crucial missing keywords.' },
  { icon: '🐙', title: 'GitHub Code Profiler', desc: 'Fetches public repos, commit languages, stars, and activity score automatically.' },
  { icon: '📋', title: '12 Industry Benchmark Roles', desc: 'Tailored benchmarks for Data Scientist, ML Engineer, Full Stack, DevOps, Cloud Architect & more.' },
]

const ROLES = [
  'Data Scientist', 'ML Engineer', 'Frontend Dev', 'Backend Dev',
  'Full Stack', 'DevOps', 'Data Analyst', 'Cloud Architect'
]

const STATS = [
  { value: '5', label: 'Custom DL Models', sub: 'PyTorch + Keras' },
  { value: '35', label: 'Engineered Features', sub: 'Pandas + NumPy' },
  { value: '5,000+', label: 'Skill Vocabulary', sub: 'spaCy NLP Pipeline' },
  { value: '12', label: 'Role Templates', sub: 'Industry Standard' },
]

export default function Landing() {
  return (
    <div className="landing">
      {/* Hero */}
      <section className="hero">
        <div className="hero-bg" />
        <div className="container">
          <div className="hero-content">
            <div className="hero-badge fade-up">
              <span className="badge badge-primary">✨ Custom AI & Deep Learning Profile Evaluator</span>
            </div>
            <h1 className="hero-title fade-up delay-1">
              Optimize Your Career with <br />
              <span className="text-gradient">Deep Learning AI</span>
            </h1>
            <p className="hero-subtitle fade-up delay-2">
              Built with custom PyTorch & Keras neural networks, spaCy NER, and NLTK — evaluate your CV,
              skills, ATS compatibility, and GitHub profile with zero guesswork.
            </p>
            <div className="hero-actions fade-up delay-3">
              <Link to="/analyze" className="btn btn-primary btn-lg">
                🚀 Analyze My CV Free
              </Link>
              <a href="#how-it-works" className="btn btn-outline btn-lg">
                How It Works
              </a>
            </div>

            {/* Role pills */}
            <div className="role-pills fade-up delay-4">
              <span className="role-label">Supported Roles:</span>
              {ROLES.map(r => (
                <span key={r} className="role-pill">{r}</span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="stats-section">
        <div className="container">
          <div className="stats-grid">
            {STATS.map(s => (
              <div key={s.label} className="stat-card fade-up">
                <div className="stat-value text-gradient">{s.value}</div>
                <div className="stat-label">{s.label}</div>
                <div className="stat-sub">{s.sub}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="hiw-section">
        <div className="container">
          <div className="section-header text-center fade-up">
            <span className="badge badge-info mb-3">AI Pipeline</span>
            <h2>How Your Profile Is Analyzed in <span className="text-gradient">4 Steps</span></h2>
            <p className="text-secondary" style={{ marginTop: 8 }}>
              Fast, transparent, and completely automated end-to-end evaluation
            </p>
          </div>
          <div className="hiw-steps">
            {[
              { n: '01', icon: '📄', title: 'Upload Resume', desc: 'Upload your PDF or DOCX file. PyMuPDF extracts raw text cleanly.' },
              { n: '02', icon: '🔍', title: 'NLP Extraction', desc: 'spaCy NER identifies contact details, skills, education & experience timeline.' },
              { n: '03', icon: '🧠', title: '5 DL Models Run', desc: 'PyTorch & Keras neural networks score quality, skill match, ATS & experience.' },
              { n: '04', icon: '📈', title: 'Transparent Report', desc: 'View your 100-pt breakdown, missing skills list, ATS warnings & recommendations.' },
            ].map((step, i) => (
              <div key={step.n} className={`hiw-step fade-up delay-${i + 1}`}>
                <div className="hiw-number">{step.n}</div>
                <div className="hiw-icon">{step.icon}</div>
                <h3>{step.title}</h3>
                <p>{step.desc}</p>
                {i < 3 && <div className="hiw-arrow">→</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features-section">
        <div className="container">
          <div className="section-header text-center fade-up">
            <span className="badge badge-primary mb-3">Capabilities</span>
            <h2>Everything You Need to <span className="text-gradient">Land Your Next Role</span></h2>
          </div>
          <div className="features-grid">
            {FEATURES.map((f, i) => (
              <div key={f.title} className={`card feature-card fade-up delay-${(i % 3) + 1}`}>
                <div className="feature-icon">{f.icon}</div>
                <h3 className="feature-title">{f.title}</h3>
                <p className="feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <div className="container">
          <div className="cta-card fade-up">
            <h2>Ready to Test Your <span className="text-gradient">Profile Score?</span></h2>
            <p>Get instant, actionable insights using custom deep learning algorithms.</p>
            <Link to="/analyze" className="btn btn-primary btn-lg">
              🎯 Analyze Resume Now
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
