"""
TouchDesigner StreamDiffusion Main Entry Point

Minimal main script leveraging the full DotSimulate StreamDiffusion fork capabilities.
Replaces the complex main_sdtd.py with a cleaner, config-driven approach.

Reads configuration from td_config.yaml (single source of truth)
"""

import logging
import os
import sys
import threading
import time

import yaml
from pythonosc import udp_client


class OSCReporter:
    """
    Lightweight OSC reporter for state, telemetry, and errors.
    Lives at module level - starts before any heavy imports.
    No dependencies on torch, diffusers, or StreamDiffusion.
    """

    def __init__(self, transmit_ip: str, transmit_port: int):
        self.client = udp_client.SimpleUDPClient(transmit_ip, transmit_port)
        self._heartbeat_running = False
        self._heartbeat_thread = None
        self._current_state = "local_offline"
        self._vram_monitor_thread = None
        self._vram_monitor_running = False

    # === Heartbeat ===
    def start_heartbeat(self) -> None:
        """Start server heartbeat immediately (Serveractive=True in TD)"""
        if self._heartbeat_running:
            return
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_running = False

    def _heartbeat_loop(self) -> None:
        while self._heartbeat_running:
            self.client.send_message("/server_active", 1)
            # Also send current state with heartbeat (every 1-2 seconds during startup)
            if self._current_state:
                self.client.send_message("/stream-info/state", self._current_state)
            time.sleep(1.0)

    # === State Management ===
    def set_state(self, state: str) -> None:
        """Set and broadcast connection state"""
        self._current_state = state
        self.client.send_message("/stream-info/state", state)

    def send_error(self, error_msg: str, error_time: int = None) -> None:
        """Send error message with timestamp"""
        if error_time is None:
            error_time = int(time.time() * 1000)
        self.client.send_message("/stream-info/error", error_msg)
        self.client.send_message("/stream-info/error-time", error_time)

    # === Telemetry (called by manager) ===
    def send_output_name(self, name: str) -> None:
        self.client.send_message("/stream-info/output-name", name)

    def send_frame_count(self, count: int) -> None:
        self.client.send_message("/framecount", count)

    def send_frame_ready(self, count: int) -> None:
        self.client.send_message("/frame_ready", count)

    def send_fps(self, fps: float) -> None:
        self.client.send_message("/stream-info/fps", fps)

    def send_controlnet_processed_name(self, name: str) -> None:
        self.client.send_message("/stream-info/controlnet-processed-name", name)

    # === VRAM Monitoring (Non-Blocking) ===
    def start_vram_monitoring(self, interval: float = 2.0) -> None:
        """Start periodic VRAM monitoring (every 2 seconds by default)"""
        if self._vram_monitor_running:
            return
        self._vram_monitor_running = True
        self._vram_monitor_thread = threading.Thread(
            target=self._vram_monitor_loop, args=(interval,), daemon=True
        )
        self._vram_monitor_thread.start()

    def stop_vram_monitoring(self) -> None:
        self._vram_monitor_running = False

    def _vram_monitor_loop(self, interval: float) -> None:
        """Periodic VRAM monitoring using torch.cuda.mem_get_info() for accurate tracking"""
        import torch

        while self._vram_monitor_running:
            try:
                if torch.cuda.is_available():
                    # Use mem_get_info() to get ACTUAL free/total VRAM (includes TensorRT!)
                    free_cuda, total_cuda = torch.cuda.mem_get_info(
                        0
                    )  # Returns (free, total) in bytes
                    total = total_cuda / (1024**3)
                    used = (total_cuda - free_cuda) / (
                        1024**3
                    )  # ACTUAL usage including TensorRT

                    # Only send useful metrics (total and used)
                    self.client.send_message("/vram/total", total)
                    self.client.send_message("/vram/used", used)

            except Exception:
                pass  # Silently skip if monitoring fails

            time.sleep(interval)

    # === Engine Building Progress ===
    def send_engine_progress(self, stage: str, model_name: str = "") -> None:
        """
        Send engine building progress.
        Stages: 'exporting_onnx', 'optimizing_onnx', 'building_engine', 'cached', 'complete'
        """
        self.client.send_message("/engine/stage", stage)
        if model_name:
            self.client.send_message("/engine/model", model_name)


class OSCLoggingHandler(logging.Handler):
    """
    Custom logging handler that sends ERROR logs to TouchDesigner via OSC.
    Also captures engine building progress messages.
    """

    def __init__(self, osc_reporter: OSCReporter):
        super().__init__()
        self.osc_reporter = osc_reporter
        # Only capture ERROR and CRITICAL
        self.setLevel(logging.ERROR)

    def emit(self, record: logging.LogRecord):
        try:
            # Check if this is a reportable error (from report_error())
            if hasattr(record, "report_error") and record.report_error:
                error_msg = self.format(record)
                self.osc_reporter.send_error(error_msg)

            # Also send all ERROR/CRITICAL logs
            elif record.levelno >= logging.ERROR:
                error_msg = self.format(record)
                self.osc_reporter.send_error(error_msg)

            # Capture engine building progress from log messages
            msg = record.getMessage()
            if "Exporting model:" in msg:
                model_name = msg.split("Exporting model:")[-1].strip()
                self.osc_reporter.send_engine_progress("exporting_onnx", model_name)
            elif "Generating optimizing model:" in msg:
                model_name = msg.split("Generating optimizing model:")[-1].strip()
                self.osc_reporter.send_engine_progress("optimizing_onnx", model_name)
            elif "Found cached engine:" in msg:
                self.osc_reporter.send_engine_progress("cached")
            elif "Building TensorRT engine" in msg or "Building engine" in msg:
                self.osc_reporter.send_engine_progress("building_engine")

        except Exception:
            self.handleError(record)


# Read config FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(script_dir, "td_config.yaml")
with open(yaml_config_path, "r") as f:
    yaml_config = yaml.safe_load(f)

td_settings = yaml_config.get("td_settings", {})
osc_port = td_settings.get("osc_receive_port", 8567)

# Create reporter and start heartbeat IMMEDIATELY
osc_reporter = OSCReporter("127.0.0.1", osc_port)
osc_reporter.start_heartbeat()  # Serveractive=True NOW in TouchDesigner
osc_reporter.start_vram_monitoring(interval=2.0)  # Monitor VRAM every 2 seconds
osc_reporter.set_state("local_starting_server")

# Add OSC logging handler to root logger (captures all logs)
osc_log_handler = OSCLoggingHandler(osc_reporter)
logging.root.addHandler(osc_log_handler)


import signal
import warnings


# Configure warnings to display in dark cyan (low saturation, dark) BEFORE any imports
def warning_format(message, category, filename, lineno, file=None, line=None):
    return f"\033[38;5;66m{filename}:{lineno}: {category.__name__}: {message}\033[0m\n"


warnings.formatwarning = warning_format

# Loading animation
print("\033[38;5;80mLoading StreamDiffusionTD", end="", flush=True)
for _ in range(3):
    time.sleep(0.33)
    print(".", end="", flush=True)
print("\033[0m")
time.sleep(0.01)

# Clear the loading line and add spacing
print("\r" + " " * 50 + "\r", end="")  # Clear line
print("\n")  # One blank line

# ASCII Art Logo
print("\033[38;5;208m     ┌──────────────────────────────────────┐")
print("     │ ▶ StreamDiffusionTD                  │")
print("     │ ▓▓▒▒░░ real-time diffusion           │")
print("     │ ░░▒▒▓▓ TOX by dotsimulate            │")
print("     │ ────────────────────────── v0.3.0    │")
print("     └──────────────────────────────────────┘")
print("     StreamDiffusion: cumulo-autumn • Daydream\033[0m\n\n")

# Add StreamDiffusion to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Suppress known third-party deprecation warnings (timm, cuda-python, controlnet_aux)
import warnings

warnings.filterwarnings(
    "ignore", message=".*timm.models.layers.*", category=FutureWarning
)
warnings.filterwarnings(
    "ignore", message=".*timm.models.registry.*", category=FutureWarning
)
warnings.filterwarnings("ignore", message=".*cuda.cudart.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*not expected and will be ignored.*")
warnings.filterwarnings(
    "ignore", message=".*Overwriting tiny_vit.*", category=UserWarning
)

# Heavy imports with error handling
try:
    print("\033[38;5;66mImporting StreamDiffusion modules...\033[0m")
    from td_manager import TouchDesignerManager
    from td_osc_handler import OSCParameterHandler

    print("\033[38;5;34m✓ Imports loaded\033[0m\n")
except ImportError as e:
    error_msg = f"Import failed: {e}"
    logging.error(error_msg)  # Also log so it appears in console
    osc_reporter.send_error(error_msg)
    osc_reporter.set_state("local_error")
    raise
except Exception as e:
    error_msg = f"Initialization failed: {e}"
    logging.error(error_msg)  # Also log so it appears in console
    osc_reporter.send_error(error_msg)
    osc_reporter.set_state("local_error")
    raise

# Print YAML config in sections with animated display
print()  # One blank line before YAML
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_config_path = os.path.join(script_dir, "td_config.yaml")

with open(yaml_config_path, "r") as f:
    yaml_content = f.read()

# Split into sections based on top-level keys or comment headers
current_section = []
lines = yaml_content.split("\n")

for line in lines:
    stripped = line.lstrip()

    # Check if this is a section break (top-level key or major comment)
    is_section_break = (
        (not stripped or stripped.startswith("#"))
        and current_section
        and any(l.strip() and not l.strip().startswith("#") for l in current_section)
    )

    if is_section_break:
        # Print accumulated section
        for section_line in current_section:
            section_stripped = section_line.lstrip()
            if not section_stripped or section_stripped.startswith("#"):
                print(f"\033[38;5;66m  {section_line}\033[0m")
            else:
                indent = len(section_line) - len(section_stripped)
                if indent == 0:
                    print(f"\033[38;5;80m  {section_line}\033[0m")
                elif indent == 2:
                    print(f"\033[38;5;75m  {section_line}\033[0m")
                elif indent == 4:
                    print(f"\033[38;5;105m  {section_line}\033[0m")
                else:
                    print(f"\033[38;5;111m  {section_line}\033[0m")
        time.sleep(0.05)  # 50ms delay between sections
        current_section = [line]
    else:
        current_section.append(line)

# Print final section
for section_line in current_section:
    section_stripped = section_line.lstrip()
    if not section_stripped or section_stripped.startswith("#"):
        print(f"\033[38;5;66m  {section_line}\033[0m")
    else:
        indent = len(section_line) - len(section_stripped)
        if indent == 0:
            print(f"\033[38;5;80m  {section_line}\033[0m")
        elif indent == 2:
            print(f"\033[38;5;75m  {section_line}\033[0m")
        elif indent == 4:
            print(f"\033[38;5;105m  {section_line}\033[0m")
        else:
            print(f"\033[38;5;111m  {section_line}\033[0m")


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes for different log levels"""

    # ANSI color codes
    GREY = "\033[90m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD_RED = "\033[91m"
    RESET = "\033[0m"

    FORMATS = {
        logging.DEBUG: GREY
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + RESET,
        logging.INFO: GREY
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + RESET,
        logging.WARNING: YELLOW
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + RESET,
        logging.ERROR: RED
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + RESET,
        logging.CRITICAL: BOLD_RED
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class StreamDiffusionTD:
    """Main application class"""

    def __init__(self, osc_reporter):
        self.osc_reporter = osc_reporter  # Store reporter reference

        # Load YAML config first to get debug_mode setting
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_config_path = os.path.join(script_dir, "td_config.yaml")

        from streamdiffusion.config import load_config

        yaml_config = load_config(yaml_config_path)

        # Get TouchDesigner-specific settings from YAML
        td_settings = yaml_config.get("td_settings", {})

        # Configure logging based on debug_mode from YAML
        debug_mode = td_settings.get("debug_mode", False)
        log_level = logging.DEBUG if debug_mode else logging.WARNING

        # Configure root logger with colored formatter
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Remove existing handlers (except our OSC handler)
        for handler in root_logger.handlers[:]:
            if not isinstance(handler, OSCLoggingHandler):
                root_logger.removeHandler(handler)

        # Add colored console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)

        # Show debug mode status if enabled
        if debug_mode:
            print("\033[38;5;208m    [DEBUG MODE ENABLED]\033[0m\n")

        input_mem = td_settings.get("input_mem_name", "input_stream")

        # Make output memory name unique (prevents conflicts & different resolutions)
        base_output_name = td_settings.get("output_mem_name", "sd_to_td")
        output_mem = f"{base_output_name}_{int(time.time())}"

        # Get OSC ports from YAML
        listen_port = td_settings.get("osc_transmit_port", 8247)  # Python listens
        transmit_port = td_settings.get("osc_receive_port", 8248)  # Python transmits

        # Initialize core manager with clean YAML config, debug flag, AND osc_reporter
        osc_reporter.set_state("local_loading_models")
        print("\033[38;5;66mLoading AI models (this may take time)...\033[0m")

        try:
            self.manager = TouchDesignerManager(
                yaml_config,
                input_mem,
                output_mem,
                debug_mode=debug_mode,
                osc_reporter=osc_reporter,  # Pass reporter to manager
            )
            print("\033[38;5;34m✓ Models loaded\033[0m")
            osc_reporter.set_state("local_ready")
        except Exception as e:
            error_msg = f"Model loading failed: {e}"
            logging.error(error_msg, exc_info=True)
            osc_reporter.send_error(error_msg)
            osc_reporter.set_state("local_error")
            raise

        # Initialize OSC handler after manager (parameters only, no heartbeat)
        self.osc_handler = OSCParameterHandler(
            manager=self.manager,
            main_app=self,  # Pass main app for shutdown handling
            listen_port=listen_port,
            transmit_port=transmit_port,
            debug_mode=debug_mode,
        )

        # Now set OSC handler in manager
        self.manager.osc_handler = self.osc_handler

        # Application shutdown state
        self.shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print()  # Extra blank line before OSC
        print(
            f"\033[38;5;80mOSC: \033[37mListen {listen_port} -> Transmit {transmit_port}\033[0m"
        )
        print(f"\033[38;5;80mMemory: \033[37m{input_mem} -> {output_mem}\033[0m")

        # Start OSC handler immediately so /stop commands work during model loading
        self.osc_handler.start()

    def start(self):
        """Start the application"""
        try:
            # OSC handler already started in __init__ so /stop works during loading

            # Send output name via reporter
            self.osc_reporter.send_output_name(self.manager.output_mem_name)

            # Auto-start streaming (matches your current main_sdtd.py behavior)
            self.manager.start_streaming()

            # Keep main thread alive
            self._wait_for_shutdown()

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.osc_reporter.send_error(f"Application error: {e}")
            raise
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        print("\n\nShutting down...")

        try:
            self.manager.stop_streaming()
            self.osc_handler.stop()
            self.osc_reporter.stop_heartbeat()
            self.osc_reporter.stop_vram_monitoring()
            print("Shutdown complete")
        except Exception as e:
            print(f"Shutdown error: {e}")

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {sig}")
        self.shutdown()
        sys.exit(0)

    def request_shutdown(self):
        """Request application shutdown (called by OSC /stop command)"""
        print("\n\033[31mStop command received via OSC\033[0m")
        self.shutdown_requested = True

        # Force immediate exit if stuck in model loading (can't gracefully shutdown)
        # Use threading to allow OSC response to be sent before exit
        def force_exit():
            time.sleep(0.1)  # Give OSC handler time to send response
            print("Forcing exit...")
            os._exit(0)  # Hard exit (bypasses cleanup but works when blocked)

        threading.Thread(target=force_exit, daemon=True).start()

    def _wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            # Keep main thread alive
            while not self.shutdown_requested:
                time.sleep(0.1)  # Check shutdown flag more frequently

        except KeyboardInterrupt:
            raise


def main():
    """Main entry point - reads from td_config.yaml"""

    # Create and start application (no longer needs stream_config.json)
    app = StreamDiffusionTD(osc_reporter)
    app.start()


if __name__ == "__main__":
    main()
