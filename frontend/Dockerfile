# Frontend Dockerfile for GoldenSignalsAI
# Multi-stage build for optimized production image

# Stage 1: Build the application
FROM node:18-alpine as builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies (using npm install instead of ci to handle version mismatches)
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Production image
FROM node:18-alpine

WORKDIR /app

# Install serve to run the static files
RUN npm install -g serve

# Copy built files from builder
COPY --from=builder /app/dist ./dist

EXPOSE 3000

# Serve the built application
CMD ["serve", "-s", "dist", "-l", "3000"]
