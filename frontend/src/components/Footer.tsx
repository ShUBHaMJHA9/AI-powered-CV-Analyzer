import { Link } from 'react-router-dom'
import './Footer.css'

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="container">
        <div className="footer-top">
          {/* Brand Info */}
          <div className="footer-brand">
            <Link to="/" className="footer-logo">
              <span className="logo-dot" />
              SMARRTIF<span className="text-gradient">&nbsp;AI</span>
            </Link>
            <p className="footer-tagline">
              Advanced AI-powered CV, GitHub & LinkedIn Profile Intelligence Evaluator.
            </p>
            <div className="made-by-badge">
              Made with <span className="heart-icon">❤️</span> by <strong className="author-name">Shubham Kumar Jha</strong>
            </div>
          </div>

          {/* Quick Links */}
          <div className="footer-links-col">
            <h4 className="footer-col-title">Navigation</h4>
            <ul className="footer-links">
              <li><Link to="/">Home</Link></li>
              <li><Link to="/analyze">Analyze Resume</Link></li>
              <li><Link to="/dashboard">Dashboard Report</Link></li>
            </ul>
          </div>

          {/* Contact Details */}
          <div className="footer-contact-col">
            <h4 className="footer-col-title">Contact & Support</h4>
            <div className="contact-details-box">
              <div className="contact-item">
                <span className="contact-icon">👨‍💻</span>
                <div>
                  <div className="contact-lbl">Developer / Creator</div>
                  <div className="contact-val">Shubham Kumar Jha</div>
                </div>
              </div>
              <div className="contact-item">
                <span className="contact-icon">📧</span>
                <div>
                  <div className="contact-lbl">Contact Email</div>
                  <a href="mailto:shubhamjha22088@gmail.com" className="contact-email-link">
                    shubhamjha22088@gmail.com
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer Bottom Bar */}
        <div className="footer-bottom">
          <p>© {new Date().getFullYear()} SMARRTIF AI. All rights reserved.</p>
          <p className="footer-credit-line">
            Engineered with ❤️ by <span>Shubham Kumar Jha</span> • Contact: <a href="mailto:shubhamjha22088@gmail.com">shubhamjha22088@gmail.com</a>
          </p>
        </div>
      </div>
    </footer>
  )
}
