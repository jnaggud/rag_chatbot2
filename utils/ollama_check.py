"""
Utilities for checking Ollama connectivity and troubleshooting.
"""

import logging
import requests
import subprocess
import platform
import sys
import os
import traceback

logger = logging.getLogger(__name__)

def is_ollama_running():
    """
    Check if the Ollama service is running.
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        # Try to connect to Ollama API
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # API check failed, try process check
        try:
            if platform.system() == "Windows":
                cmd = "tasklist | findstr ollama"
            else:  # macOS and Linux
                cmd = "ps aux | grep '[o]llama'"  # The brackets prevent the grep command itself from showing up
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return bool(result.stdout.strip())
        except Exception as e:
            logger.error(f"Error checking Ollama process: {e}")
            return False

def test_ollama_model(model_name="llama3"):
    """
    Test if a specific Ollama model is working.
    
    Args:
        model_name: Name of the model to test
        
    Returns:
        dict: Dictionary with 'success' flag and 'message' explaining result
    """
    try:
        if not is_ollama_running():
            return {
                "success": False,
                "message": "Ollama service is not running. Please start it with 'ollama serve' in a terminal."
            }
            
        # Check if model exists locally
        check_cmd = f"ollama list | grep {model_name}"
        check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        
        if not check_result.stdout.strip():
            return {
                "success": False,
                "message": f"Model '{model_name}' is not available. Try pulling it with 'ollama pull {model_name}'."
            }
            
        # Test simple generation
        test_cmd = f"ollama run {model_name} 'Hello, is this working?' --nowordwrap"
        test_result = subprocess.run(
            test_cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=10  # Set a timeout to avoid hanging
        )
        
        if test_result.returncode != 0:
            return {
                "success": False,
                "message": f"Model test failed: {test_result.stderr}"
            }
            
        return {
            "success": True,
            "message": f"Model '{model_name}' is working properly.",
            "response": test_result.stdout.strip()
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "message": f"Model test timed out. The model might be loading or there might be a problem."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error testing model: {str(e)}"
        }

def get_troubleshooting_info():
    """
    Gather information for troubleshooting Ollama issues.
    
    Returns:
        dict: Dictionary with troubleshooting information
    """
    info = {
        "ollama_running": False,
        "os_info": platform.platform(),
        "python_version": sys.version,
        "error_logs": [],
        "environment": {}
    }
    
    try:
        # Check if Ollama is running
        info["ollama_running"] = is_ollama_running()
        
        # Get Ollama version if available
        try:
            version_cmd = "ollama --version"
            version_result = subprocess.run(version_cmd, shell=True, capture_output=True, text=True)
            info["ollama_version"] = version_result.stdout.strip()
        except Exception as e:
            info["ollama_version"] = f"Error getting version: {str(e)}"
        
        # Get available models
        try:
            models_cmd = "ollama list"
            models_result = subprocess.run(models_cmd, shell=True, capture_output=True, text=True)
            info["available_models"] = models_result.stdout.strip()
        except Exception as e:
            info["available_models"] = f"Error getting models: {str(e)}"
        
        # Get relevant environment variables
        ollama_env_vars = {k: v for k, v in os.environ.items() if 'OLLAMA' in k.upper()}
        info["environment"] = ollama_env_vars
        
    except Exception as e:
        info["error"] = str(e)
        info["traceback"] = traceback.format_exc()
    
    return info

def suggest_fixes(info):
    """
    Suggest fixes based on troubleshooting information.
    
    Args:
        info: Troubleshooting information from get_troubleshooting_info()
        
    Returns:
        list: List of suggested fixes
    """
    suggestions = []
    
    if not info["ollama_running"]:
        suggestions.append("Ollama service is not running. Start it with 'ollama serve' in a new terminal.")
    
    if "available_models" in info and "Error" in info["available_models"]:
        suggestions.append("There was an error retrieving model information. Make sure Ollama is installed correctly.")
    elif "available_models" in info and not info["available_models"]:
        suggestions.append("No models are available. Pull a model with 'ollama pull llama3'.")
    
    if "CUDA_VISIBLE_DEVICES" in info["environment"] and info["environment"]["CUDA_VISIBLE_DEVICES"] == "-1":
        suggestions.append("GPU is disabled by environment variables. Consider removing CUDA_VISIBLE_DEVICES=-1.")
    
    if not suggestions:
        suggestions.append("No obvious issues detected. Try restarting the Ollama service.")
    
    return suggestions

if __name__ == "__main__":
    # This can be run directly for diagnostics
    print("Checking Ollama status...")
    print(f"Is Ollama running? {is_ollama_running()}")
    
    print("\nTesting default model...")
    test_result = test_ollama_model()
    print(f"Success: {test_result['success']}")
    print(f"Message: {test_result['message']}")
    
    print("\nTroubleshooting info:")
    info = get_troubleshooting_info()
    for key, value in info.items():
        if key != "environment":
            print(f"{key}: {value}")
    
    print("\nSuggested fixes:")
    for suggestion in suggest_fixes(info):
        print(f"- {suggestion}")