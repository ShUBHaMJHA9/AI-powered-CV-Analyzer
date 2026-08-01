# ── Frontend Multi-Stage Dockerfile ────────────────────────────
# Stage 1: Build static React Vite bundle
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Pass Backend domain target to Vite build
ARG VITE_API_URL=https://airesume.codetechfoundation.tech
ENV VITE_API_URL=$VITE_API_URL

RUN npx vite build

# Stage 2: Serve static files with Nginx
FROM nginx:alpine

# Remove default nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
