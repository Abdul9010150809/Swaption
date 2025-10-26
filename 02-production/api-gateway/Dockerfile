# Use a lightweight Node runtime as the base image
FROM node:18-alpine

# Set working directory
WORKDIR /usr/src/app

# Install dependencies (use package-lock / npm ci for reproducible installs)
COPY package*.json ./
RUN npm ci --omit=dev

# Copy application source
COPY . .

# Build if a build script is present
RUN npm run build --if-present

# Expose default port (adjust if your app uses a different port)
EXPOSE 3000

# Start the app using the start script; adjust if your entrypoint differs
CMD ["npm", "start"]
