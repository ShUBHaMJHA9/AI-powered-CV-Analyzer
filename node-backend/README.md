# 🟢 SMARRTIF AI — Node.js API Gateway Backend

The **Node.js API Gateway** for **SMARRTIF AI** acts as the middleware bridge between the React frontend and the Python AI Inference Microservice. It handles multipart file uploads (`Multer`), CORS security, payload validation, and request proxying.

> **Made with ❤️ by [Shubham Kumar Jha](mailto:shubhamjha22088@gmail.com)**  
> ✉️ **Contact:** `shubhamjha22088@gmail.com`

---

## 🛠️ Tech Stack & Dependencies

- **Runtime:** Node.js v18+
- **Framework:** Express v4
- **File Uploads:** Multer (10MB file limit for `.pdf`, `.docx`, `.doc`)
- **HTTP Client:** Axios & FormData
- **CORS:** CORS middleware with credential support

---

## 📁 Node Backend Directory Structure

```
node-backend/
├── README.md               # Node Backend Documentation
├── package.json            # Dependencies & Scripts
├── uploads/                # Temporary disk storage for uploaded resumes
└── src/
    └── app.js              # Main Express Server & Route Definitions
```

---

## 🔌 API Route Specifications

| Method | Route | Description | Backend Handler |
|---|---|---|---|
| `GET` | `/health` | Gateway health check endpoint & developer metadata | Returns `{ status: 'ok', author: 'Shubham Kumar Jha', contact_email: '...' }` |
| `POST` | `/api/parse-cv` | Fast CV upload parser for initial URL auto-extraction | Accepts `cv` file → Proxies to Python `/parse-cv` → Returns raw text length, GitHub & LinkedIn URLs |
| `POST` | `/api/analyze` | Main CV analysis trigger | Accepts `cv` file, `target_role`, `github_username`, `linkedin_data` → Proxies to Python `/analyze` |
| `GET` | `/api/github/:username` | Standalone GitHub profile evaluation | Proxies to Python `/github/:username` |

---

## ⚙️ Environment Variables

Create a `.env` file in `node-backend/` (or copy `.env.example`):

```env
PORT=5000
FRONTEND_URL=http://localhost:5173
AI_SERVICE_URL=http://localhost:8000
```

---

## 🚀 Execution & Commands

### 1. Install Dependencies
```bash
npm install
```

### 2. Start Gateway Server
```bash
npm start
```
The Express Gateway server will run on **http://localhost:5000**.

---

## ✉️ Author & Contact

- **Developer:** Shubham Kumar Jha
- **Email:** [shubhamjha22088@gmail.com](mailto:shubhamjha22088@gmail.com)
- **Credit:** *Made with ❤️ by Shubham Kumar Jha*
