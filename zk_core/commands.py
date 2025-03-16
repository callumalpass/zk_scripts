"""
Command execution utilities for ZK Core.

This module provides utilities for running external commands with standardized
error handling and output processing.
"""

import os
import logging
import subprocess
from typing import List, Optional, Tuple, Dict, Any, Union

logger = logging.getLogger(__name__)

class CommandExecutor:
    """Class for executing external commands with proper error handling."""
    
    @staticmethod
    def run(
        cmd: List[str], 
        input_data: Optional[str] = None, 
        check: bool = False,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> Tuple[int, str, str]:
        """
        Run a command and return the return code, stdout, and stderr.
        
        Args:
            cmd: Command to run as a list of strings
            input_data: Optional string to pass as stdin
            check: Whether to raise an exception on non-zero return code
            env: Optional environment variables to set
            cwd: Optional working directory
            
        Returns:
            Tuple containing (return_code, stdout, stderr)
        """
        try:
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
                
            if input_data is not None:
                proc = subprocess.run(
                    cmd, 
                    input=input_data,
                    text=True,
                    capture_output=True,
                    check=check,
                    env=process_env,
                    cwd=cwd
                )
            else:
                proc = subprocess.run(
                    cmd,
                    text=True, 
                    capture_output=True,
                    check=check,
                    env=process_env,
                    cwd=cwd
                )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(cmd)}")
            return e.returncode, e.stdout or "", e.stderr or ""
        except Exception as e:
            logger.error(f"Error running command {' '.join(cmd)}: {e}")
            return 1, "", str(e)
    
    @staticmethod
    def run_and_check(
        cmd: List[str],
        error_message: str = "Command failed",
        input_data: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None
    ) -> Optional[str]:
        """
        Run a command, check for errors, and return stdout if successful.
        
        Args:
            cmd: Command to run as a list of strings
            error_message: Custom error message prefix
            input_data: Optional string to pass as stdin
            env: Optional environment variables to set
            cwd: Optional working directory
            
        Returns:
            Command output on success, None on failure (with error logged)
        """
        rc, stdout, stderr = CommandExecutor.run(cmd, input_data, False, env, cwd)
        if rc != 0:
            logger.error(f"{error_message}: {stderr}")
            return None
        return stdout
    
    @staticmethod
    def check_exists(command: str) -> bool:
        """
        Check if a command exists in the PATH.
        
        Args:
            command: Name of the command to check
            
        Returns:
            True if command exists, False otherwise
        """
        return CommandExecutor.run_and_check(["which", command]) is not None
    
    @staticmethod
    def find_command(possible_commands: List[str]) -> Optional[str]:
        """
        Find the first available command from a list of possible commands.
        
        Args:
            possible_commands: List of command names to check
            
        Returns:
            First available command or None if none found
        """
        for cmd in possible_commands:
            if CommandExecutor.check_exists(cmd):
                return cmd
        return None


# Legacy alias for backward compatibility
def run_command(cmd: List[str], input_data: Optional[str] = None, check: bool = False) -> Tuple[int, str, str]:
    """
    Legacy wrapper around CommandExecutor.run() for backward compatibility.
    
    This function will eventually be deprecated. Use CommandExecutor class directly.
    """
    return CommandExecutor.run(cmd, input_data, check)