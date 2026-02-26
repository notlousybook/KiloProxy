# Kilo Proxy

A fully OpenAI-compatible API proxy for Kilo.

## Features

- **Full OpenAI API compatibility** - chat completions, completions, embeddings, models
- **All OpenAI parameters supported** - max_tokens, temperature, tools, stream, response_format, etc.
- **Streaming SSE support** - real-time streaming responses
- **Background server mode** - run as a daemon process
- **Token authentication management** - easy auth configuration
- **Cross-platform** - Windows, macOS, Linux

## Installation

```bash
pip install kilo-proxy
```

Or install from source:

```bash
pip install -e .
```

## CLI Commands

### Authentication

Set your Kilo authentication token:

```bash
kilo-proxy auth [token]
```

If no token is provided, anonymous access is used by default.

### Start Server (Background)

Start the server in background mode:

```bash
kilo-proxy start [--host HOST] [--port PORT]
```

Options:
- `--host` - Host to bind to (default: 127.0.0.1)
- `--port` - Port to listen on (default: 5380)

### Stop Server

Stop the background server:

```bash
kilo-proxy stop
```

### Run Server (Foreground)

Run the server in foreground mode:

```bash
kilo-proxy proxy [--host HOST] [--port PORT]
```

Options:
- `--host` - Host to bind to (default: 127.0.0.1)
- `--port` - Port to listen on (default: 5380)

### Server Status

Check the current server status:

```bash
kilo-proxy status
```

### List Models

List all available models:

```bash
kilo-proxy models
```

### Broke Mode (Toggle Free Models)

Toggle broke mode (only show free models):

```bash
kilo-proxy broke          # Toggle broke mode
kilo-proxy broke on       # Enable broke mode
kilo-proxy broke off      # Disable broke mode
kilo-proxy broke --list   # List all free models
```

### Restart Server

Restart the proxy server:

```bash
kilo-proxy restart
```

### Generate New Session

Generate a new session ID to bypass rate limits:

```bash
kilo-proxy new-session
```

### View Logs

View server logs:

```bash
kilo-proxy logs                    # Show last 100 lines
kilo-proxy logs --lines 50        # Show last 50 lines
kilo-proxy logs --errors          # Show errors only
kilo-proxy logs --follow          # Follow log output
```

### Start Proxy and OpenCode

Start the proxy and launch OpenCode:

```bash
kilo-proxy opencode
```

### IP Shuffler

Rotate your IP address to bypass rate limits! Load proxies from a URL or file, and the shuffler will automatically rotate your IP at a set interval.

```bash
# Enable/disable IP shuffling
kilo-proxy shuffle on
kilo-proxy shuffle off

# Add proxies manually
kilo-proxy shuffle add http://ip:port
kilo-proxy shuffle add socks5://ip:port

# Load proxies from URL or file (one per line, # for comments)
kilo-proxy shuffle load https://example.com/proxies.txt
kilo-proxy shuffle load ./proxies.txt

# List configured proxies
kilo-proxy shuffle list

# Set shuffle interval (minimum 60 seconds)
kilo-proxy shuffle interval 300

# Force immediate shuffle
kilo-proxy shuffle now

# Check all proxies (mark working/broken)
kilo-proxy shuffle check
kilo-proxy shuffle check --remove  # Auto-remove broken proxies

# Fix/normalize proxy URLs
kilo-proxy shuffle fix

# View shuffler status
kilo-proxy shuffle status
```

### Configuration Wizard

Interactive configuration:

```bash
kilo-proxy config         # Run configuration wizard
kilo-proxy config --show  # Show current config
```

### Install OpenCode Integration

Install and configure OpenCode to use Kilo models:

```bash
kilo-proxy install opencode
```

This interactive wizard will:
1. Find or create your OpenCode config
2. Ask if you're broke (free models only)
3. Let you select models for all or per agent/category
4. Optionally install Oh My OpenCode support
5. Map Oh My OpenCode models to Kilo models
6. Write the configuration files

### Show Configuration

Display the current configuration:

```bash
kilo-proxy config-show
```

### Autolaunch (Run on Boot)

Install the proxy to start automatically on system boot:

```bash
kilo-proxy autolaunch
```

Remove the proxy from boot:

```bash
kilo-proxy unautolaunch
```

Works on Windows (Task Scheduler), macOS (LaunchAgents), and Linux (systemd user services).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/models` | List all available models |
| GET | `/v1/models/{model_id}` | Get a specific model |
| POST | `/v1/chat/completions` | Create a chat completion |
| POST | `/v1/completions` | Create a text completion |
| POST | `/v1/embeddings` | Create embeddings |
| GET | `/v1/engines` | List engines (legacy) |
| GET | `/health` | Health check endpoint |

## Usage Examples

### List Models

```bash
curl http://localhost:5380/v1/models
```

### Chat Completion (Non-streaming)

```bash
curl http://localhost:5380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-ai/glm-5:free",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat Completion (Streaming)

```bash
curl http://localhost:5380/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "z-ai/glm-5:free",
    "messages": [
      {"role": "user", "content": "Tell me a short story"}
    ],
    "stream": true,
    "max_tokens": 500
  }'
```

## Configuration

| File | Location |
|------|----------|
| Config file | `~/.kilo-proxy/config.json` |
| PID file | `~/.kilo-proxy/server.pid` |

## Supported Models

Popular free models available through Kilo:

- `z-ai/glm-5:free`
- `minimax/minimax-m2.5:free`
- `deepseek/deepseek-chat:free`
- `meta-llama/llama-3-8b-instruct:free`
- `qwen/qwen-2-7b-instruct:free`

Run `kilo-proxy models` to see all available models.

## Requirements

- Python 3.9+

## License

MIT

## Author

- **lousybook94** (notlousybook)
- GitHub: https://github.com/notlousybook/KiloProxy
- PyPI: https://pypi.org/project/kilo-proxy/
- Email: lousybook94@gmail.com
- Location: Pluto, the planet
