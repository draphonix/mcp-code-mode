FROM python:3.11-slim

# Install Node.js and npm (required for MCP servers using npx)
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set up user to avoid running as root (HF requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# Copy project files
COPY --chown=user . .

# Use the HF-specific MCP servers config
COPY --chown=user mcp_servers_hf.json mcp_servers.json

# Install dependencies
# Note: We install from the current directory to include the package itself if needed
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
