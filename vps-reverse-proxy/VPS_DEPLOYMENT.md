# 🚀 VPS Deployment & Domain Nginx Reverse Proxy Guide

This step-by-step guide explains how to deploy **SMARRTIF AI** on a Linux VPS (Ubuntu/Debian) using **Docker Compose**, **Nginx Reverse Proxy**, and **Let's Encrypt SSL Certbot** connected to your custom domain.

> **Made with ❤️ by [Shubham Kumar Jha](mailto:shubhamjha22088@gmail.com)**  
> ✉️ **Contact:** `shubhamjha22088@gmail.com`

---

## 📋 Prerequisites on VPS

1. **VPS Specs:** Ubuntu 20.04/22.04 LTS (Minimum 2GB RAM recommended).
2. **Domain Name:** DNS `A` record pointing your domain (e.g. `cv-analyzer.yourdomain.com`) to your VPS IP address.
3. **Ports Open:** 80 (HTTP) & 443 (HTTPS) allowed in firewall (`ufw allow 80/tcp`, `ufw allow 443/tcp`).

---

## ⚡ Step 1: Install Docker & Docker Compose on VPS

SSH into your VPS and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git nginx certbot python3-certbot-nginx

sudo systemctl enable --now docker
sudo systemctl enable --now nginx
```

---

## 📥 Step 2: Clone Repository & Build Docker Containers

```bash
git clone https://github.com/ShUBHaMJHA9/AI-powered-CV-Analyzer.git
cd AI-powered-CV-Analyzer

# Build and start all 3 microservices in detached mode
sudo docker-compose up -d --build
```

Check running containers:
```bash
sudo docker-compose ps
```

---

## 🌐 Step 3: Configure Nginx Reverse Proxy for Your Domain

Copy the Nginx VPS config to `/etc/nginx/sites-available/`:

```bash
sudo cp vps-reverse-proxy/nginx-vps.conf /etc/nginx/sites-available/smarrtif-ai
```

Edit the file and replace `yourdomain.com` with your actual domain name:

```bash
sudo nano /etc/nginx/sites-available/smarrtif-ai
```

Link the configuration & reload Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/smarrtif-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔒 Step 4: Enable Free HTTPS/SSL with Certbot

Run Certbot to automatically issue and install SSL certificates for your domain:

```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

Certbot will automatically update `/etc/nginx/sites-available/smarrtif-ai` to redirect HTTP traffic to HTTPS securely!

---

## ✅ Step 5: Verification & Useful Commands

- **Check API Gateway:** `https://yourdomain.com/api/health`
- **Check AI Engine:** `https://yourdomain.com/ai-health`
- **View Container Logs:** `sudo docker-compose logs -f`
- **Restart All Services:** `sudo docker-compose restart`
- **Update Application:**
  ```bash
  git pull
  sudo docker-compose up -d --build
  ```
