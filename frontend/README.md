# ⚛️ SMARRTIF AI — Frontend Web Application

The frontend user interface for **SMARRTIF AI** is a modern, responsive web application built with **React 19**, **TypeScript**, and **Vite**. It features an interactive resume analysis wizard, real-time score gauges, Recharts visual analytics, and an enterprise light-theme design system.

> **Made with ❤️ by [Shubham Kumar Jha](mailto:shubhamjha22088@gmail.com)**  
> ✉️ **Contact:** `shubhamjha22088@gmail.com`

---

## 🎨 Design System & UI Architecture

- **Theme & Aesthetics:** Enterprise Light Modern Theme inspired by Vercel, Linear, and Stripe. Built with CSS tokens in `src/index.css`.
- **Typography:** Plus Jakarta Sans & Inter from Google Fonts.
- **Charts & Visualizations:** `Recharts` library for 6-axis Radar Charts and GitHub Language Distribution Pie Charts.
- **Interactive SVG Icons:** Color-coded badges for Target Role 🎯, Overall Score 📊, Radar 🕸️, GitHub 🐙, LinkedIn 💼, ATS 🤖, Skills ⚡, and Experience 📈.

---

## 📁 Frontend Directory Structure

```
frontend/
├── README.md               # Frontend Documentation
├── index.html              # HTML Entry Point
├── package.json            # Node Dependencies & Scripts
├── tsconfig.json           # TypeScript Configuration
├── vite.config.ts          # Vite Server Configuration
└── src/
    ├── main.tsx            # React Root Entry Point
    ├── App.tsx             # Main Router & Layout
    ├── index.css           # Global Theme Tokens & Styles
    ├── components/
    │   ├── Navbar.tsx      # Top Navigation Header
    │   ├── Footer.tsx      # Creator Credit & Contact Footer
    │   └── Footer.css      # Footer Styles
    └── pages/
        ├── Landing.tsx     # Hero Landing Page & Features
        ├── Landing.css
        ├── Analyze.tsx     # 4-Step Resume Upload & Integration Wizard
        ├── Analyze.css
        ├── Dashboard.tsx   # AI Evaluation Report & Recharts Dashboard
        └── Dashboard.css
```

---

## 🚀 Pages & User Flow

### 1. Landing Page (`/`)
- Hero section introducing **SMARRTIF AI**.
- Feature cards showcasing PyTorch, Keras, spaCy, GitHub, and LinkedIn analysis.
- Live Call to Action to start resume evaluation.

### 2. Analysis Wizard (`/analyze`)
- **Step 1 — Resume Upload:** Drag-and-drop zone for PDF and DOCX files. Triggers quick auto-extraction of GitHub & LinkedIn URLs.
- **Step 2 — Target Role Selection:** Choose from 12+ tech roles (Data Scientist, ML Engineer, Full Stack, Cloud Architect, etc.).
- **Step 3 — Profile Integrations:** Input/review GitHub username, LinkedIn URL, headline, college degree, and certifications.
- **Step 4 — Review & Run:** Overview of selected parameters and active DL models before triggering the AI pipeline.

### 3. Dashboard Report (`/dashboard`)
- **Score Gauge:** Radial animated gauge with overall score and grade pill (`A+`, `A`, `B+`, etc.).
- **Dimension Progress Grid:** 6 progress meters for CV Quality, Skill Match, ATS Rating, Experience, GitHub, and LinkedIn.
- **Profile Radar Chart:** 6-axis Recharts Radar Chart visualizing candidate strengths.
- **GitHub Developer Panel:** Avatar, bio, repos count, stars count ⭐, followers 👥, and language breakdown pie chart.
- **LinkedIn Intelligence Panel:** Verified handle, headline, college/education tag 🎓, certifications list 📜, and 4-bar quality metric breakdown.
- **ATS Screening Engine:** Pass probability meter, risk status badge, and missing keyword pills.
- **Skills Heatmap:** 2-column comparison of skills present vs missing skills.
- **AI Action Recommendations:** Prioritized action items sorted by impact.

---

## 🛠️ Installation & Execution

### 1. Install Dependencies
```bash
npm install
```

### 2. Run Local Development Server
```bash
npm run dev
```
Access the frontend at **http://localhost:5173**.

### 3. Build Production Bundle
```bash
npm run build
```

---

## ✉️ Author & Contact

- **Developer:** Shubham Kumar Jha
- **Email:** [shubhamjha22088@gmail.com](mailto:shubhamjha22088@gmail.com)
- **Credit:** *Made with ❤️ by Shubham Kumar Jha*
