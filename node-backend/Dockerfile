# ── Node.js API Gateway Dockerfile ─────────────────────────────
FROM node:18-alpine

WORKDIR /app

# Copy package files & install production dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application source
COPY . .

# Ensure uploads directory exists
RUN mkdir -p uploads

# Expose API Gateway port
EXPOSE 5000

# Set environment defaults
ENV PORT=5000
ENV NODE_ENV=production
ENV AI_SERVICE_URL=http://python-ai:8000

# Run Express server
CMD ["node", "src/app.js"]
