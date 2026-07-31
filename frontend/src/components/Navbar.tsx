import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <Link to="/" className="navbar-logo">
          <span className="logo-dot" />
          SMARRTIF<span className="text-gradient">&nbsp;AI</span>
        </Link>
        <div className="navbar-links">
          <Link to="/" className={`navbar-link ${pathname === '/' ? 'active' : ''}`}>Home</Link>
          <Link to="/analyze" className={`navbar-link ${pathname === '/analyze' ? 'active' : ''}`}>Analyze</Link>
          <Link to="/analyze" className="btn btn-primary btn-sm">
            🚀 Get Started
          </Link>
        </div>
      </div>
    </nav>
  )
}
