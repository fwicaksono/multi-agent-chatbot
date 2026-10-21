# syntax=docker/dockerfile:1.7
# ---- Build the static SPA ---------------------------------------------------
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# ---- Serve with nginx; /api is reverse-proxied to the FastAPI service --------
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=10s --timeout=3s CMD wget -qO- http://localhost/healthz || exit 1
