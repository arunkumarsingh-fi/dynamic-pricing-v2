#!/usr/bin/env python3
"""
🚀 Full Circle Exchange - One-Click Deployment Script
====================================================

This script handles everything needed to get the Dynamic Pricing V2 application
running on your machine, regardless of your technical background.

Just run: python3 code_deployment.py

What this script does:
- Checks and installs Podman if needed
- Installs podman-compose if needed  
- Sets up the data pipeline
- Builds and deploys containers
- Ensures networking works correctly
- Verifies everything is running properly

Author: Full Circle Exchange Team
"""

import os
import sys
import subprocess
import time
import json
import platform
import urllib.request
import urllib.error
from pathlib import Path

class Colors:
    """Terminal colors for pretty output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class FullCircleDeployment:
    def __init__(self):
        self.os_type = platform.system().lower()
        self.is_mac = self.os_type == 'darwin'
        self.is_linux = self.os_type == 'linux'
        self.home_dir = Path.home()
        self.current_dir = Path.cwd()
        
    def print_header(self):
        """Print fancy header"""
        print(f"{Colors.CYAN}{Colors.BOLD}")
        print("=" * 60)
        print("🚀 FULL CIRCLE EXCHANGE - AUTO DEPLOYMENT")
        print("   Dynamic Pricing V2 - AI Asset Optimization")
        print("=" * 60)
        print(f"{Colors.END}")
        
    def print_step(self, step_num, title, description=""):
        """Print step with nice formatting"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}STEP {step_num}: {title}{Colors.END}")
        if description:
            print(f"{Colors.WHITE}   {description}{Colors.END}")
            
    def print_success(self, message):
        """Print success message"""
        print(f"{Colors.GREEN}✅ {message}{Colors.END}")
        
    def print_warning(self, message):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")
        
    def print_error(self, message):
        """Print error message"""
        print(f"{Colors.RED}❌ {message}{Colors.END}")
        
    def print_info(self, message):
        """Print info message"""
        print(f"{Colors.CYAN}ℹ️  {message}{Colors.END}")
        
    def run_command(self, cmd, shell=True, capture_output=True):
        """Run command with error handling"""
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
                return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
            else:
                result = subprocess.run(cmd, shell=shell)
                return result.returncode == 0, "", ""
        except Exception as e:
            return False, "", str(e)
            
    def check_python(self):
        """Check Python version"""
        self.print_step(1, "Checking Python Environment")
        
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.print_error("Python 3.8 or higher is required")
            self.print_info("Please install Python 3.8+ and try again")
            sys.exit(1)
        
        self.print_success(f"Python {python_version.major}.{python_version.minor} detected")
        
    def check_and_install_podman(self):
        """Check if Podman is installed and install if needed"""
        self.print_step(2, "Checking Podman Installation", "Container runtime required for deployment")
        
        # Check if podman is already installed
        success, _, _ = self.run_command("podman --version")
        if success:
            self.print_success("Podman is already installed")
            return
            
        self.print_warning("Podman not found. Installing Podman...")
        
        if self.is_mac:
            # Check if Homebrew is installed
            success, _, _ = self.run_command("brew --version")
            if not success:
                self.print_error("Homebrew is required to install Podman on macOS")
                self.print_info("Install Homebrew first: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                sys.exit(1)
                
            print("   Installing Podman via Homebrew (this may take a few minutes)...")
            success, stdout, stderr = self.run_command("brew install podman", capture_output=False)
            if not success:
                self.print_error(f"Failed to install Podman: {stderr}")
                sys.exit(1)
                
        elif self.is_linux:
            # Try different package managers
            success = False
            
            # Try apt (Ubuntu/Debian)
            if not success:
                apt_success, _, _ = self.run_command("which apt-get")
                if apt_success:
                    print("   Installing Podman via apt...")
                    success, _, stderr = self.run_command("sudo apt-get update && sudo apt-get install -y podman", capture_output=False)
                    
            # Try dnf (Fedora/RHEL)
            if not success:
                dnf_success, _, _ = self.run_command("which dnf")
                if dnf_success:
                    print("   Installing Podman via dnf...")
                    success, _, stderr = self.run_command("sudo dnf install -y podman", capture_output=False)
                    
            # Try pacman (Arch)
            if not success:
                pacman_success, _, _ = self.run_command("which pacman")
                if pacman_success:
                    print("   Installing Podman via pacman...")
                    success, _, stderr = self.run_command("sudo pacman -S --noconfirm podman", capture_output=False)
                    
            if not success:
                self.print_error("Could not install Podman automatically")
                self.print_info("Please install Podman manually for your Linux distribution")
                sys.exit(1)
        else:
            self.print_error(f"Unsupported operating system: {self.os_type}")
            self.print_info("Please install Podman manually")
            sys.exit(1)
            
        # Verify installation
        success, version, _ = self.run_command("podman --version")
        if success:
            self.print_success(f"Podman installed successfully: {version}")
        else:
            self.print_error("Podman installation failed")
            sys.exit(1)
            
    def setup_podman_machine(self):
        """Initialize Podman machine on macOS"""
        if not self.is_mac:
            return
            
        self.print_step(3, "Setting up Podman Machine", "Required for Podman on macOS")
        
        # Check if podman machine is already running
        success, output, _ = self.run_command("podman machine list")
        if success and "running" in output.lower():
            self.print_success("Podman machine is already running")
            return
            
        # Initialize podman machine if it doesn't exist
        success, output, _ = self.run_command("podman machine list")
        if success and not output.strip():
            print("   Initializing Podman machine (this may take a few minutes)...")
            success, _, stderr = self.run_command("podman machine init", capture_output=False)
            if not success:
                self.print_error(f"Failed to initialize Podman machine: {stderr}")
                sys.exit(1)
                
        # Start podman machine
        print("   Starting Podman machine...")
        success, _, stderr = self.run_command("podman machine start", capture_output=False)
        if not success:
            self.print_warning("Podman machine may already be running")
            
        # Verify machine is running
        time.sleep(3)
        success, _, _ = self.run_command("podman info")
        if success:
            self.print_success("Podman machine is running")
        else:
            self.print_error("Failed to start Podman machine")
            sys.exit(1)
            
    def install_podman_compose(self):
        """Install podman-compose"""
        self.print_step(4, "Installing podman-compose", "Container orchestration tool")
        
        # Check if already installed
        success, _, _ = self.run_command("podman-compose --version")
        if success:
            self.print_success("podman-compose is already installed")
            return
            
        print("   Installing podman-compose...")
        success, _, stderr = self.run_command(f"{sys.executable} -m pip install podman-compose")
        if not success:
            self.print_error(f"Failed to install podman-compose: {stderr}")
            sys.exit(1)
            
        # Verify installation
        success, version, _ = self.run_command("podman-compose --version")
        if success:
            self.print_success(f"podman-compose installed: {version}")
        else:
            self.print_error("podman-compose installation failed")
            sys.exit(1)
            
    def setup_data_pipeline(self):
        """Generate demo data and setup data pipeline"""
        self.print_step(5, "Setting up Data Pipeline", "Generating demo data for AI training")
        
        # Create data directory
        data_dir = self.current_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Generate demo data if not exists
        analytics_file = data_dir / "analytics_data.csv"
        if not analytics_file.exists():
            print("   Generating synthetic transaction data (10k records)...")
            success, stdout, stderr = self.run_command(f"{sys.executable} data_simulator.py")
            if not success:
                self.print_error(f"Failed to generate demo data: {stderr}")
                sys.exit(1)
            self.print_success("Transaction data generated successfully")
        else:
            self.print_success("Transaction data already exists")
            
        # Ensure ETL data files exist (will be generated by containers)
        self.print_success("Data pipeline setup complete")
        
    def clean_old_containers(self):
        """Clean up any old containers"""
        self.print_step(6, "Cleaning Up", "Removing old containers and networks")
        
        # Stop and remove old containers
        self.run_command("podman-compose -f podman-compose.yml down")
        self.run_command("podman container prune -f")
        
        # Wait a moment for cleanup
        time.sleep(2)
        self.print_success("Cleanup complete")
        
    def build_and_deploy_containers(self):
        """Build and deploy the application containers"""
        self.print_step(7, "Building and Deploying", "This may take 5-10 minutes on first run")
        
        print("   Building container images...")
        success, stdout, stderr = self.run_command("podman-compose -f podman-compose.yml build", capture_output=False)
        if not success:
            self.print_error("Failed to build containers")
            self.print_error(stderr)
            sys.exit(1)
            
        print("   Starting containers...")
        success, stdout, stderr = self.run_command("podman-compose -f podman-compose.yml up -d")
        if not success:
            self.print_error("Failed to start containers")
            self.print_error(stderr)
            sys.exit(1)
            
        self.print_success("Containers built and started")
        
    def wait_for_services(self):
        """Wait for services to be ready"""
        self.print_step(8, "Waiting for Services", "Ensuring all components are ready")
        
        max_attempts = 30
        ml_ready = False
        ui_ready = False
        
        for attempt in range(max_attempts):
            print(f"   Checking services... ({attempt + 1}/{max_attempts})")
            
            # Check ML API
            if not ml_ready:
                try:
                    response = urllib.request.urlopen("http://localhost:5003/health", timeout=5)
                    if response.status == 200:
                        ml_ready = True
                        print(f"   {Colors.GREEN}✅ ML API is ready{Colors.END}")
                except:
                    pass
                    
            # Check UI
            if not ui_ready:
                try:
                    response = urllib.request.urlopen("http://localhost:8503", timeout=5)
                    if response.status == 200:
                        ui_ready = True
                        print(f"   {Colors.GREEN}✅ UI is ready{Colors.END}")
                except:
                    pass
                    
            if ml_ready and ui_ready:
                break
                
            time.sleep(5)
            
        if not (ml_ready and ui_ready):
            self.print_warning("Services may still be starting up")
            self.print_info("Check logs with: podman-compose -f podman-compose.yml logs")
        else:
            self.print_success("All services are ready!")
            
    def run_data_initialization(self):
        """Ensure all data is initialized before UI becomes usable"""
        self.print_step(9, "Initializing Data", "Preparing AI models and demo datasets")
        
        # Wait a bit more for containers to fully initialize
        print("   Allowing containers to fully initialize...")
        time.sleep(10)
        
        # Check if ML container has generated its data files
        success, stdout, stderr = self.run_command("podman exec full-circle-ml-v2 ls -la /app/data/")
        if success:
            self.print_success("ML data files are ready")
        else:
            self.print_warning("ML container may still be initializing data")
            
        # Verify UI can connect to ML API
        try:
            # Test ML API health
            response = urllib.request.urlopen("http://localhost:5003/health", timeout=10)
            if response.status == 200:
                self.print_success("ML API is responding correctly")
            else:
                self.print_warning("ML API health check returned non-200 status")
        except Exception as e:
            self.print_warning(f"ML API not yet ready: {e}")
            
        self.print_success("Data initialization complete")
        
    def verify_deployment(self):
        """Verify the deployment is working"""
        self.print_step(10, "Verifying Deployment", "Final health checks")
        
        # Check container status
        success, output, _ = self.run_command("podman ps --filter name=full-circle")
        if success and "full-circle-ml-v2" in output and "full-circle-ui-v2" in output:
            self.print_success("All containers are running")
        else:
            self.print_warning("Some containers may not be running properly")
            
        # Final service checks
        ml_ok = False
        ui_ok = False
        
        try:
            response = urllib.request.urlopen("http://localhost:5003/health", timeout=5)
            ml_ok = response.status == 200
        except:
            pass
            
        try:
            response = urllib.request.urlopen("http://localhost:8503", timeout=5)
            ui_ok = response.status == 200
        except:
            pass
            
        if ml_ok and ui_ok:
            self.print_success("All services verified successfully")
            return True
        else:
            self.print_warning("Some services may need more time to start")
            return False
            
    def print_success_message(self):
        """Print final success message with access information"""
        print(f"\n{Colors.GREEN}{Colors.BOLD}")
        print("=" * 60)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 60)
        print(f"{Colors.END}")
        
        print(f"{Colors.CYAN}📱 Access your Full Circle Exchange application:{Colors.END}")
        print(f"   🌐 {Colors.BOLD}Web Interface: http://localhost:8503{Colors.END}")
        print(f"   🔧 {Colors.BOLD}ML API: http://localhost:5003{Colors.END}")
        
        print(f"\n{Colors.YELLOW}🛠️  Management Commands:{Colors.END}")
        print(f"   Stop:    {Colors.WHITE}./deploy.sh stop{Colors.END}")
        print(f"   Restart: {Colors.WHITE}./deploy.sh restart{Colors.END}")
        print(f"   Logs:    {Colors.WHITE}./deploy.sh logs{Colors.END}")
        print(f"   Status:  {Colors.WHITE}./deploy.sh status{Colors.END}")
        
        print(f"\n{Colors.MAGENTA}💡 Quick Start Guide:{Colors.END}")
        print(f"   1. Open {Colors.BOLD}http://localhost:8503{Colors.END} in your browser")
        print(f"   2. Go to '📱 Story of a Unit' tab")
        print(f"   3. Select an iPhone model and set parameters")
        print(f"   4. Click 'Single Market Analysis' to see AI recommendations")
        print(f"   5. Explore other tabs for business intelligence")
        
        print(f"\n{Colors.BLUE}🧠 What You Get:{Colors.END}")
        print("   • AI-powered pricing recommendations")
        print("   • Multi-market optimization")
        print("   • Real-time business analytics")
        print("   • Machine learning that improves with feedback")
        
        print(f"\n{Colors.WHITE}Ready to optimize your iPhone resale business!{Colors.END}\n")
        
    def print_troubleshooting(self):
        """Print troubleshooting information if deployment fails"""
        print(f"\n{Colors.YELLOW}{Colors.BOLD}TROUBLESHOOTING GUIDE{Colors.END}")
        print(f"{Colors.YELLOW}=" * 40 + f"{Colors.END}")
        
        print(f"\n{Colors.RED}If you see connection errors:{Colors.END}")
        print("   1. Check container status: podman ps")
        print("   2. View logs: podman-compose -f podman-compose.yml logs")
        print("   3. Restart services: ./deploy.sh restart")
        
        print(f"\n{Colors.RED}If containers won't start:{Colors.END}")
        print("   1. Clean up: podman-compose -f podman-compose.yml down")
        print("   2. Remove old images: podman image prune -a")
        print("   3. Run this script again")
        
        print(f"\n{Colors.RED}If Podman issues on macOS:{Colors.END}")
        print("   1. Restart Podman machine: podman machine stop && podman machine start")
        print("   2. Reset if needed: podman machine rm && podman machine init")
        
        print(f"\n{Colors.CYAN}Need help? Check:{Colors.END}")
        print("   • README.md for detailed documentation")
        print("   • DEPLOYMENT_GUIDE.md for advanced configuration")
        
    def deploy(self):
        """Main deployment function"""
        try:
            self.print_header()
            
            # Run deployment steps
            self.check_python()
            self.check_and_install_podman()
            self.setup_podman_machine()
            self.install_podman_compose()
            self.setup_data_pipeline()
            self.clean_old_containers()
            self.build_and_deploy_containers()
            self.wait_for_services()
            self.run_data_initialization()
            
            # Verify and show results
            if self.verify_deployment():
                self.print_success_message()
            else:
                self.print_troubleshooting()
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Deployment cancelled by user{Colors.END}")
            sys.exit(1)
        except Exception as e:
            self.print_error(f"Deployment failed: {e}")
            self.print_troubleshooting()
            sys.exit(1)

def main():
    """Main entry point"""
    deployment = FullCircleDeployment()
    deployment.deploy()

if __name__ == "__main__":
    main()
