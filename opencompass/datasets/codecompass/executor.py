# isort: skip_file
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional


class LocalExecutor:
    """Executes code locally with custom time and memory limits.

    Refined to distinguish between Runtime and Memory Limit errors.
    """

    def __init__(self,
                 timeout: int = 10,
                 memory_limit_mb: int = 512,
                 temp_base_dir: Optional[str] = None):
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

        if temp_base_dir:
            self.temp_base_dir = Path(temp_base_dir)
            self.temp_base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_base_dir = Path(tempfile.gettempdir())

    def _set_resource_limits(self):
        # isort: off
        import resource
        # isort: on
        """This function is called in the child process right before exec."""
        try:
            mem_bytes = self.memory_limit_mb * 1024 * 1024
            # Set both soft and hard limits for virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass

    def _compile_cpp(self, source_file: Path, temp_dir: Path) -> tuple:
        """Compiles C++ source code."""
        output_file = temp_dir / source_file.stem
        cmd = [
            'g++', '-std=c++17', '-O2',
            str(source_file), '-o',
            str(output_file)
        ]
        try:
            proc = subprocess.run(cmd,
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            if proc.returncode == 0:
                return True, '', output_file
            else:
                return False, proc.stderr, None
        except subprocess.TimeoutExpired:
            return False, 'Compilation timed out', None
        except Exception as e:
            return False, f'Compilation error: {str(e)}', None

    def _run_executable(self, exec_file: Path, stdin_data: str) -> Dict:
        """Runs the compiled executable and interprets the result."""
        start_time = time.time()

        try:
            # The preexec_fn is the key for setting limits on the child process
            proc = subprocess.run([str(exec_file)],
                                  input=stdin_data,
                                  capture_output=True,
                                  text=True,
                                  timeout=self.timeout,
                                  preexec_fn=self._set_resource_limits)
            execution_time = time.time() - start_time

            if proc.returncode == 0:

                status = {'id': 3, 'description': 'Accepted'}
            elif proc.returncode < 0:

                status = {'id': 5, 'description': 'Memory Limit Exceeded'}
            else:

                status = {'id': 11, 'description': 'Runtime Error'}

            return {
                'status': status,
                'stdout': proc.stdout,
                'stderr': proc.stderr,
                'time': execution_time
            }

        except subprocess.TimeoutExpired:
            # Wall-clock time limit exceeded
            return {
                'status': {
                    'id': 5,
                    'description': 'Time Limit Exceeded'
                },
                'stdout': '',
                'stderr': f'Timeout > {self.timeout}s',
                'time': self.timeout
            }
        except Exception as e:
            return {
                'status': {
                    'id': 11,
                    'description': 'Runtime Error'
                },
                'stdout': '',
                'stderr': str(e),
                'time': time.time() - start_time
            }

    def execute_code(self, source_code: str, stdin: str, language: str,
                     temp_dir: Path) -> Dict:
        """Orchestrates compilation and execution."""
        if language.lower() not in ['c++', 'g++']:
            return {
                'status': {
                    'id': -1,
                    'description': 'Unsupported Language'
                },
                'stderr': f'Language {language} not supported.'
            }

        # Create a unique file to avoid race conditions in parallel execution
        source_file = temp_dir / f'solution_{os.getpid()}_{time.time_ns()}.cpp'
        with open(source_file, 'w') as f:
            f.write(source_code)

        success, err, exec_file = self._compile_cpp(source_file, temp_dir)
        if not success:
            return {
                'status': {
                    'id': 6,
                    'description': 'Compilation Error'
                },
                'compile_output': err
            }

        return self._run_executable(exec_file, stdin)

    def verify_output(self, result: Dict, expected_output: str) -> Dict:
        """Verifies the stdout against the expected output."""
        # Only verify if the execution was successful
        if result.get('status', {}).get('description') != 'Accepted':
            return result

        # Normalize whitespace and line endings for robust comparison
        stdout = result.get('stdout', '').replace('\r\n', '\n').strip()
        expected = expected_output.replace('\r\n', '\n').strip()

        # Trailing whitespace on each line can also be an issue
        stdout_lines = [line.rstrip() for line in stdout.split('\n')]
        expected_lines = [line.rstrip() for line in expected.split('\n')]

        if stdout_lines == expected_lines:
            result['status'] = {'id': 3, 'description': 'Accepted'}
        else:
            result['status'] = {'id': 4, 'description': 'Wrong Answer'}
        return result

    def submit_code(self,
                    source_code: str,
                    stdin: str,
                    expected_output: str,
                    language: str = 'C++') -> Dict:
        """Public entry point.

        Manages temp directories and orchestrates the full process.
        """
        with tempfile.TemporaryDirectory(
                dir=self.temp_base_dir) as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            result = self.execute_code(source_code, stdin, language, temp_dir)

            # The result from execute_code can be TLE, MLE, RTE, etc.
            # We only verify the output if the status is "Accepted" initially.
            if result['status']['description'] == 'Accepted':
                result = self.verify_output(result, expected_output)

            return result
