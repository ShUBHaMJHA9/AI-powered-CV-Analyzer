import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from 'recharts'
import './Dashboard.css'

const COLORS = ['#4F46E5', '#0EA5E9', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899']

function ScoreGauge({ score }: { score: number }) {
  const [displayed, setDisplayed] = useState(0)
  useEffect(() => {
    let start = 0
    const timer = setInterval(() => {
      start += 2
      if (start >= score) { setDisplayed(score); clearInterval(timer) }
      else setDisplayed(start)
    }, 15)
    return () => clearInterval(timer)
  }, [score])

  const color = score >= 75 ? '#10B981' : score >= 55 ? '#F59E0B' : '#EF4444'
  const dash = 2 * Math.PI * 56
  const filled = (displayed / 100) * dash

  return (
    <div className="gauge-wrapper">
      <svg width="170" height="170" viewBox="0 0 170 170">
        <circle cx="85" cy="85" r="56" fill="none" stroke="#E2E8F0" strokeWidth="12" />
        <circle
          cx="85" cy="85" r="56" fill="none"
          stroke={color} strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${dash - filled}`}
          strokeDashoffset={dash / 4}
          style={{ transition: 'stroke-dasharray 0.05s linear', filter: `drop-shadow(0 2px 6px ${color}44)` }}
        />
      </svg>
      <div className="gauge-center">
        <div className="gauge-score" style={{ color }}>{displayed}</div>
        <div className="gauge-label">out of 100</div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [result, setResult] = useState<any>(null)

  useEffect(() => {
    const raw = localStorage.getItem('cv_result')
    if (!raw) { navigate('/analyze'); return }
    setResult(JSON.parse(raw))
  }, [navigate])

  if (!result) return <div className="flex-center" style={{ height: '80vh' }}><div className="spinner" /></div>

  const { overall_score, grade, target_role, dimensions, nlp_analysis, skill_gaps, top_recommendations } = result

  const radarData = [
    { subject: 'CV Quality', score: (dimensions.cv_quality?.weighted_score / 30) * 100 || 0 },
    { subject: 'Skill Match', score: (dimensions.skill_match?.weighted_score / 25) * 100 || 0 },
    { subject: 'ATS Rating', score: (dimensions.ats_compatibility?.weighted_score / 15) * 100 || 0 },
    { subject: 'Experience', score: (dimensions.experience_level?.weighted_score / 20) * 100 || 0 },
    { subject: 'GitHub Score', score: (dimensions.github_profile?.weighted_score / 10) * 100 || 0 },
    { subject: 'LinkedIn Score', score: (dimensions.linkedin_profile?.weighted_score / 10) * 100 || 0 },
  ]

  const githubData = result.github_data || dimensions.github_profile
  const linkedinData = result.linkedin_data || dimensions.linkedin_profile
  const topLanguages = Object.entries(result.github_languages || githubData?.top_languages || {}).slice(0, 5)
  const present = dimensions.skill_match?.skills_present || []
  const missing = skill_gaps || dimensions.skill_match?.skills_missing || []

  const gradeColors: Record<string, string> = { 'A+': '#10B981', 'A': '#10B981', 'B+': '#0EA5E9', 'B': '#F59E0B', 'C': '#F59E0B', 'D': '#EF4444' }
  const gradeColor = gradeColors[String(grade)] || '#4F46E5'

  return (
    <div className="dashboard-page">
      <div className="container">
        {/* Header */}
        <div className="dash-header fade-up">
          <div className="dash-title-group">
            <div className="role-tag-pill">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="m12 8 4 4-4 4M8 12h8"/></svg>
              <span>{target_role?.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())}</span>
            </div>
            <h1>AI Resume Evaluation <span className="text-gradient">Report</span></h1>
            <p className="dash-subtitle">Multi-dimensional analysis powered by PyTorch, Keras, spaCy & NLTK</p>
          </div>
          <div className="dash-actions">
            <Link to="/analyze" className="btn btn-outline btn-sm">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/></svg>
              New Analysis
            </Link>
          </div>
        </div>

        {/* Top Row: Main Score Card + Radar Chart */}
        <div className="top-row">
          {/* Main Overall Score Card */}
          <div className="card score-card fade-up">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-indigo">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
              </div>
              <h3 className="card-title-text">Overall Evaluation Score</h3>
            </div>
            <div className="score-main-flex">
              <ScoreGauge score={overall_score} />
              <div className="score-details-col">
                <div className="grade-badge-lg" style={{ color: gradeColor, borderColor: gradeColor, background: `${gradeColor}10` }}>
                  Grade: {grade}
                </div>
                <div className="score-summary-txt">
                  {overall_score >= 80 ? '🌟 Exceptional candidate profile. Highly competitive across all dimensions.' :
                   overall_score >= 65 ? '👍 Strong resume with solid core skills and experience match.' :
                   '⚡ Action needed: Follow AI recommendations to boost your interview callbacks.'}
                </div>
              </div>
            </div>

            {/* Dimension Breakdown Bars */}
            <div className="dim-bars-grid">
              {[
                { label: 'CV Quality', score: dimensions.cv_quality?.weighted_score || 0, max: 30, color: '#4F46E5', icon: '📄' },
                { label: 'Skill Match', score: dimensions.skill_match?.weighted_score || 0, max: 25, color: '#0EA5E9', icon: '⚡' },
                { label: 'ATS Rating', score: dimensions.ats_compatibility?.weighted_score || 0, max: 15, color: '#10B981', icon: '🤖' },
                { label: 'Experience', score: dimensions.experience_level?.weighted_score || 0, max: 20, color: '#F59E0B', icon: '📈' },
                { label: 'GitHub Profile', score: dimensions.github_profile?.weighted_score || 0, max: 10, color: '#8B5CF6', icon: '🐙' },
                { label: 'LinkedIn Profile', score: dimensions.linkedin_profile?.weighted_score || 0, max: 10, color: '#0077B5', icon: '💼' },
              ].map(d => (
                <div key={d.label} className="dim-item-card">
                  <div className="dim-item-top">
                    <span className="dim-item-label">{d.icon} {d.label}</span>
                    <span className="dim-item-pts">{d.score.toFixed(1)} <small>/ {d.max}</small></span>
                  </div>
                  <div className="dim-progress-track">
                    <div className="dim-progress-fill" style={{ width: `${Math.min(100, (d.score / d.max) * 100)}%`, background: d.color }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Radar Chart Card */}
          <div className="card radar-card fade-up delay-1">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-cyan">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/><path d="M12 12 2.1 12"/><path d="M12 12 19 5"/></svg>
              </div>
              <h3 className="card-title-text">Profile Dimension Radar</h3>
            </div>
            <div className="radar-container">
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData} outerRadius="70%">
                  <PolarGrid stroke="#CBD5E1" strokeDasharray="3 3" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: '#334155', fontSize: 12, fontWeight: 600 }} />
                  <Radar name="Candidate Score" dataKey="score" stroke="#4F46E5" fill="#4F46E5" fillOpacity={0.25} strokeWidth={2.5} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* GitHub & LinkedIn Row */}
        <div className="two-col">
          {/* GitHub Intelligence Panel */}
          <div className="card gh-panel-card fade-up">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-purple">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              </div>
              <h3 className="card-title-text">GitHub Developer Profile</h3>
              {githubData?.username && (
                <span className="badge badge-primary style-right">@{githubData.username}</span>
              )}
            </div>

            {githubData && !githubData.error ? (
              <div className="gh-body">
                <div className="gh-user-banner">
                  {githubData.avatar_url && (
                    <img src={githubData.avatar_url} alt="GitHub Avatar" className="gh-avatar" />
                  )}
                  <div>
                    <h4 className="gh-user-name">{githubData.name || githubData.username}</h4>
                    <p className="gh-user-bio">{githubData.bio || 'Public GitHub Developer Profile'}</p>
                  </div>
                </div>

                <div className="gh-metrics-grid">
                  <div className="gh-metric-box">
                    <span className="gh-metric-num">{githubData.public_repos || 0}</span>
                    <span className="gh-metric-lbl">Public Repos</span>
                  </div>
                  <div className="gh-metric-box">
                    <span className="gh-metric-num">⭐ {githubData.total_stars || 0}</span>
                    <span className="gh-metric-lbl">Stars Received</span>
                  </div>
                  <div className="gh-metric-box">
                    <span className="gh-metric-num">👥 {githubData.followers || 0}</span>
                    <span className="gh-metric-lbl">Followers</span>
                  </div>
                  <div className="gh-metric-box">
                    <span className="gh-metric-num">{dimensions.github_profile?.weighted_score}/10</span>
                    <span className="gh-metric-lbl">GitHub Score</span>
                  </div>
                </div>

                {topLanguages.length > 0 && (
                  <div className="gh-langs-container">
                    <h5 className="sub-section-title">Top Languages & Tech Distribution</h5>
                    <div className="lang-pie-wrapper">
                      <ResponsiveContainer width={160} height={160}>
                        <PieChart>
                          <Pie data={topLanguages.map(([name, pct]) => ({ name, value: pct }))} cx={80} cy={80} innerRadius={45} outerRadius={68} dataKey="value">
                            {topLanguages.map((_, idx) => <Cell key={idx} fill={COLORS[idx % COLORS.length]} />)}
                          </Pie>
                          <Tooltip formatter={(v: any) => `${v}%`} />
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="lang-legend-list">
                        {topLanguages.map(([lang, pct], idx) => (
                          <div key={lang} className="lang-legend-item">
                            <span className="lang-dot" style={{ background: COLORS[idx % COLORS.length] }} />
                            <span className="lang-name">{lang}</span>
                            <span className="lang-pct">{String(pct)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-profile-box">
                <div className="empty-icon">🐙</div>
                <p className="empty-txt">{githubData?.reason || githubData?.error || 'No GitHub username provided.'}</p>
                <Link to="/analyze" className="btn btn-outline btn-sm">Add GitHub Profile →</Link>
              </div>
            )}
          </div>

          {/* LinkedIn Intelligence Panel */}
          <div className="card linkedin-panel-card fade-up delay-1">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-blue">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
              </div>
              <h3 className="card-title-text">LinkedIn Profile Analysis</h3>
              {dimensions.linkedin_profile?.weighted_score > 0 && (
                <span className="badge badge-success style-right">{dimensions.linkedin_profile.weighted_score}/10 Pts</span>
              )}
            </div>

            {linkedinData && (linkedinData.has_url || linkedinData.headline || linkedinData.url || linkedinData.education) ? (
              <div className="linkedin-body">
                <div className="linkedin-card-box">
                  <div className="linkedin-head-row">
                    <span className="linkedin-badge-tag">Verified LinkedIn Intelligence</span>
                    {linkedinData.scraped && <span className="badge badge-info">Scraped Metadata</span>}
                  </div>
                  {linkedinData.url && (
                    <a href={linkedinData.url} target="_blank" rel="noopener noreferrer" className="linkedin-url-link">
                      {linkedinData.url} 🔗
                    </a>
                  )}
                  {linkedinData.headline && (
                    <div className="linkedin-headline-box">
                      <span className="headline-lbl">Headline & Current Title:</span>
                      <p className="headline-txt">"{linkedinData.headline}"</p>
                    </div>
                  )}
                  {linkedinData.education && (
                    <div className="linkedin-edu-box" style={{ marginTop: 8 }}>
                      <span className="headline-lbl">🎓 College / Education:</span>
                      <p className="headline-txt">{linkedinData.education}</p>
                    </div>
                  )}
                  {linkedinData.certifications && linkedinData.certifications.length > 0 && (
                    <div className="linkedin-certs-box" style={{ marginTop: 8 }}>
                      <span className="headline-lbl">📜 Certifications & Licenses:</span>
                      <div className="kw-tags-flex" style={{ marginTop: 4 }}>
                        {linkedinData.certifications.map((c: string, idx: number) => (
                          <span key={idx} className="skill-chip chip-present">{c}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="linkedin-scores-breakdown">
                  <h5 className="sub-section-title">LinkedIn Quality Metrics</h5>
                  <div className="linkedin-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
                    <div className="lk-item">
                      <span className="lk-lbl">Headline Match</span>
                      <span className="lk-val">{dimensions.linkedin_profile?.breakdown?.headline_match || 0} / 3.0</span>
                    </div>
                    <div className="lk-item">
                      <span className="lk-lbl">Skills Match</span>
                      <span className="lk-val">{dimensions.linkedin_profile?.breakdown?.skills_alignment || 0} / 2.5</span>
                    </div>
                    <div className="lk-item">
                      <span className="lk-lbl">College & Edu</span>
                      <span className="lk-val">{dimensions.linkedin_profile?.breakdown?.education_college || 0} / 2.5</span>
                    </div>
                    <div className="lk-item">
                      <span className="lk-lbl">Certifications</span>
                      <span className="lk-val">{dimensions.linkedin_profile?.breakdown?.certifications || 0} / 1.0</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-profile-box">
                <div className="empty-icon">💼</div>
                <p className="empty-txt">{linkedinData?.reason || 'No LinkedIn profile URL attached to CV.'}</p>
                <Link to="/analyze" className="btn btn-outline btn-sm">Attach LinkedIn URL →</Link>
              </div>
            )}
          </div>
        </div>

        {/* ATS Screening Engine */}
        <div className="card ats-panel-card fade-up">
          <div className="card-header-bar">
            <div className="card-icon-badge icon-emerald">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="m9 12 2 2 4-4"/></svg>
            </div>
            <h3 className="card-title-text">ATS (Applicant Tracking System) Screening</h3>
            <span className={`badge style-right ${dimensions.ats_compatibility?.ats_pass ? 'badge-success' : 'badge-danger'}`}>
              {dimensions.ats_compatibility?.ats_pass ? '✓ High ATS Pass Probability' : '⚠️ ATS Filtering Risk'}
            </span>
          </div>

          <div className="ats-row-grid">
            <div className="ats-metric-summary">
              <div className="ats-big-num" style={{ color: dimensions.ats_compatibility?.ats_score >= 60 ? '#10B981' : '#EF4444' }}>
                {dimensions.ats_compatibility?.ats_score?.toFixed(0) || 0}%
              </div>
              <div className="ats-big-lbl">BiLSTM Keyword Match Probability</div>
            </div>
            <div className="ats-keywords-container">
              <h5 className="sub-section-title">Critical ATS Keywords to Add:</h5>
              <div className="kw-tags-flex">
                {(dimensions.ats_compatibility?.keywords_missing || []).length > 0 ? (
                  dimensions.ats_compatibility.keywords_missing.slice(0, 10).map((kw: string) => (
                    <span key={kw} className="kw-pill">+ {kw}</span>
                  ))
                ) : (
                  <span className="badge badge-success">Great! All critical ATS keywords detected.</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Skills Analysis */}
        <div className="two-col">
          <div className="card fade-up">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-emerald">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
              </div>
              <h3 className="card-title-text">Skills Matched for Role ({present.length})</h3>
            </div>
            <div className="skill-tags-grid">
              {present.length > 0 ? present.map((s: string) => (
                <span key={s} className="skill-chip chip-present">
                  ✓ {s}
                </span>
              )) : <p className="text-muted">No specific role skills detected.</p>}
            </div>
          </div>

          <div className="card fade-up delay-1">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-amber">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              </div>
              <h3 className="card-title-text">Recommended Skill Additions ({missing.length})</h3>
            </div>
            <div className="skill-tags-grid">
              {missing.length > 0 ? missing.map((s: string) => (
                <span key={s} className="skill-chip chip-missing">
                  + {s}
                </span>
              )) : <p className="text-muted">No skill gaps found!</p>}
            </div>
          </div>
        </div>

        {/* Experience Classifier */}
        <div className="card fade-up">
          <div className="card-header-bar">
            <div className="card-icon-badge icon-amber">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 20v-6"/><path d="M6 20V10"/><path d="M18 20V4"/></svg>
            </div>
            <h3 className="card-title-text">Seniority & Experience Detector (Keras LSTM)</h3>
          </div>
          <div className="exp-grid-cards">
            <div className="exp-box">
              <span className="exp-box-lbl">Detected Seniority</span>
              <span className="exp-box-val">{dimensions.experience_level?.detected_level || 'N/A'}</span>
            </div>
            <div className="exp-box">
              <span className="exp-box-lbl">Role Requirement</span>
              <span className="exp-box-val">{dimensions.experience_level?.required_level || 'N/A'}</span>
            </div>
            <div className="exp-box">
              <span className="exp-box-lbl">Total Experience</span>
              <span className="exp-box-val">{dimensions.experience_level?.detected_years ?? 'N/A'} Yrs</span>
            </div>
            <div className="exp-box">
              <span className="exp-box-lbl">Role Seniority Match</span>
              <div className="exp-box-val">
                {dimensions.experience_level?.level_match
                  ? <span className="badge badge-success">✓ Meets Requirements</span>
                  : <span className="badge badge-danger">⚠️ Below Target Level</span>}
              </div>
            </div>
          </div>
        </div>

        {/* AI Recommendations */}
        {top_recommendations?.length > 0 && (
          <div className="card fade-up">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-indigo">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
              </div>
              <h3 className="card-title-text">Prioritized AI Action Recommendations</h3>
            </div>
            <div className="recs-wrapper">
              {top_recommendations.map((rec: any, i: number) => (
                <div key={i} className="rec-card-row">
                  <div className="rec-num-badge">#{i + 1}</div>
                  <div className="rec-body-content">
                    <div className="rec-action-title">{rec.action}</div>
                    <span className={`badge ${rec.impact === 'High' ? 'badge-danger' : 'badge-warning'}`}>
                      {rec.impact} Priority
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Extracted Resume Data (spaCy NER) */}
        {nlp_analysis?.contact && (
          <div className="card fade-up">
            <div className="card-header-bar">
              <div className="card-icon-badge icon-cyan">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
              </div>
              <h3 className="card-title-text">Extracted Candidate Resume Metadata</h3>
            </div>
            <div className="contact-meta-grid">
              {[
                { label: 'Candidate Name', value: nlp_analysis.contact.name },
                { label: 'Email Address', value: nlp_analysis.contact.email },
                { label: 'Phone Number', value: nlp_analysis.contact.phone },
                { label: 'LinkedIn Handle', value: nlp_analysis.contact.linkedin_url },
                { label: 'GitHub Handle', value: nlp_analysis.contact.github_url },
                { label: 'Detected Skills Count', value: `${nlp_analysis.skills?.total_count || 0} skills categorized` },
              ].map(({ label, value }) => value && (
                <div key={label} className="meta-info-box">
                  <span className="meta-box-lbl">{label}</span>
                  <span className="meta-box-val">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
