import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Landing from './pages/Landing'
import Analyze from './pages/Analyze'
import Dashboard from './pages/Dashboard'
import './index.css'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <div className="page-wrapper">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
      <Footer />
    </BrowserRouter>
  )
}
