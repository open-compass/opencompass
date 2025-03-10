import os
import subprocess
import uuid
import yaml
import argparse
from typing import Dict, Optional
from dataclasses import dataclass, field, asdict
from loguru import logger

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    "volcano_deploy_{time}.log",
    rotation="500 MB",
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    lambda msg: print(msg, flush=True),  # Also print to console
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
)

def get_current_conda_env() -> str:
    """Get the path of the current conda environment."""
    try:
        # Get CONDA_PREFIX from environment
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            logger.debug(f"Found conda environment from CONDA_PREFIX: {conda_prefix}")
            return conda_prefix
        
        # If CONDA_PREFIX is not set, try to get it from conda command
        logger.debug("CONDA_PREFIX not found, trying conda info command")
        result = subprocess.run(
            'conda info --envs | grep "*" | awk \'{print $NF}\'',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            env_path = result.stdout.strip()
            logger.debug(f"Found conda environment from command: {env_path}")
            return env_path
        

        
            
    except Exception as e:
        logger.warning(f"Failed to detect conda environment: {e}")
    
    # Return default if detection fails
    default_env = '/fs-computility/llm/shared/llmeval/share_envs/oc-v034-ld-v061'
    logger.warning(f"Using default conda environment: {default_env}")
    return default_env

@dataclass
class VolcanoConfig:
    """Configuration for Volcano deployment."""
    home_path = '/fs-computility/llmeval/zhaoyufeng/'
    bashrc_path=f'{home_path}.bashrc'
    conda_env_name: str = field(default_factory=get_current_conda_env)
    huggingface_cache: str = '/fs-computility/llm/shared/llmeval/models/opencompass_hf_hub'
    torch_cache: str = '/fs-computility/llm/shared/llmeval/torch'
    volc_cfg_file: str = '/fs-computility/llmeval/zhaoyufeng/ocplayground/envs/volc_infer.yaml'

    task_name: str = 'compassjudger-1-32B'
    queue_name: str = 'llmit'
    extra_envs: list = field(default_factory=lambda: [
        'COMPASS_DATA_CACHE=/fs-computility/llm/shared/llmeval/datasets/compass_data_cache',
        'TORCH_HOME=/fs-computility/llm/shared/llmeval/torch',
        'TIKTOKEN_CACHE_DIR=/fs-computility/llm/shared/llmeval/share_tiktoken',
    ])
    image: str = "vemlp-cn-beijing.cr.volces.com/preset-images/cuda:12.2.2"

class VolcanoDeployment:
    """Handles deployment of ML tasks to Volcano infrastructure."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize deployment with configuration."""
        self.config = VolcanoConfig(**config) if config else VolcanoConfig()
        self.pwd = os.getcwd()
        logger.info("Initialized VolcanoDeployment with configuration:")
        logger.info(f"Working directory: {self.pwd}")
        for key, value in asdict(self.config).items():
            logger.info(f"{key}: {value}")
        
    def choose_flavor(self, num_gpus: int, num_replicas: int = 1) -> Dict:
        """Select appropriate machine flavor based on GPU requirements."""
        flavor_map = {
            0: 'ml.c1ie.2xlarge',
            1: 'ml.pni2l.3xlarge',
            2: 'ml.pni2l.7xlarge',
            4: 'ml.pni2l.14xlarge',
            8: 'ml.hpcpni2l.28xlarge'
        }
        
        if num_gpus > 8:
            logger.error(f"Configuration for {num_gpus} GPUs not supported")
            raise NotImplementedError(f"Configuration for {num_gpus} GPUs not supported")
        
        for max_gpus, flavor in sorted(flavor_map.items()):
            if num_gpus <= max_gpus:
                selected_flavor = flavor
                break
        
        logger.info(f"Selected flavor {selected_flavor} for {num_gpus} GPUs")
        logger.info(f"Number of relicas: {num_replicas}")
                
        with open(self.config.volc_cfg_file) as fp:
            volc_cfg = yaml.safe_load(fp)
            
        for role_spec in volc_cfg['TaskRoleSpecs']:
            if role_spec['RoleName'] == 'worker':
                role_spec['Flavor'] = selected_flavor
                role_spec['RoleReplicas'] = num_replicas
                
        return volc_cfg
    
    def build_shell_command(self, task_cmd: str) -> str:
        """Construct shell command with all necessary environment setup."""
        logger.debug("Building shell command")
        cmd_parts = [
            f'source {self.config.bashrc_path}',
        ]
        
        # Get CONDA_EXE from enviroment
        conda_exe = os.environ.get("CONDA_EXE", None)
        assert conda_exe, f"CONDA_EXE is None, please make sure conda exists in your current environment"
        conda_activate = conda_exe.replace("bin/conda", "bin/activate")

        # Handle conda environment activation based on whether it's a path or name
        if os.path.exists(self.config.conda_env_name):
            logger.debug(f"Using conda activate with path: {self.config.conda_env_name}")
            cmd_parts.append(f'source {conda_activate} {self.config.conda_env_name}')
        else:
            logger.debug(f"Using source activate with name: {self.config.conda_env_name}")
            cmd_parts.append(f'source {conda_activate} {self.config.conda_env_name}')
        
        cmd_parts.extend([
            f'export PYTHONPATH={self.pwd}:$PYTHONPATH',
            f'export HF_HUB_CACHE={self.config.huggingface_cache}',
            f'export HUGGINGFACE_HUB_CACHE={self.config.huggingface_cache}',
            f'export TORCH_HOME={self.config.torch_cache}'
        ])
        
        offline_vars = [
            'HF_DATASETS_OFFLINE=1',
            'TRANSFORMERS_OFFLINE=1',
            'HF_EVALUATE_OFFLINE=1',
            'HF_HUB_OFFLINE=1'
        ]
        cmd_parts.extend([f'export {var}' for var in offline_vars])
        
        if self.config.extra_envs:
            cmd_parts.extend([f'export {env}' for env in self.config.extra_envs])
        
        cmd_parts.extend([
            f'cd {self.pwd}',
            task_cmd
        ])
        
        full_cmd = '; '.join(cmd_parts)
        logger.debug(f"Generated shell command: {full_cmd}")
        return full_cmd
    
    def deploy(self, task_cmd: str, num_gpus: int = 4,  num_replicas: int = 1) -> subprocess.CompletedProcess:
        """Deploy the task to Volcano infrastructure."""
        logger.info(f"Starting deployment with {num_gpus} GPUs")
        logger.info(f"Task command: {task_cmd}")
        
        try:
            volcano_cfg = self.choose_flavor(num_gpus, num_replicas)
            
            os.makedirs(f'{self.pwd}/tmp', exist_ok=True)
            tmp_cfg_file = f'{self.pwd}/tmp/{uuid.uuid4()}_cfg.yaml'
            logger.debug(f"Created temporary config file: {tmp_cfg_file}")
            
            with open(tmp_cfg_file, 'w') as fp:
                yaml.dump(volcano_cfg, fp, sort_keys=False)
            
            shell_cmd = self.build_shell_command(task_cmd)
            
            submit_cmd = (
                'volc ml_task submit'
                f" --conf '{tmp_cfg_file}'"
                f" --entrypoint '{shell_cmd}'"
                f' --task_name {self.config.task_name}'
                f' --resource_queue_name {self.config.queue_name}'
                f' --image {self.config.image}'
            )
            
            logger.info("Submitting Volcano task")
            logger.debug(f"Submit command: {submit_cmd}")
            
            result = subprocess.run(
                submit_cmd,
                shell=True,
                text=True,
                capture_output=True,
                check=True
            )
            
            logger.info("Task submitted successfully")
            return result
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise
        finally:
            pass
            # if os.path.exists(tmp_cfg_file):
            #     logger.debug(f"Cleaning up temporary config file: {tmp_cfg_file}")
            #     os.remove(tmp_cfg_file)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy ML tasks to Volcano infrastructure')
    
    # Required arguments
    parser.add_argument('--task-cmd', required=True, help='The main task command to execute')
    
    # Optional arguments
    parser.add_argument('--num-gpus', type=int, default=4, help='Number of GPUs required (default: 4)')
    parser.add_argument('--num-replicas', type=int, default=1, help='Number of Replicas (default: 1)')
    parser.add_argument('--task-name', help='Override default task name')
    parser.add_argument('--queue-name', help='Override default queue name')
    parser.add_argument('--image', help="Overide default image")
    parser.add_argument('--conda-env', help='Conda environment to use (default: current environment)')
    parser.add_argument('--extra-envs', nargs='+', help='Additional environment variables in format KEY=VALUE')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                      help='Set logging level (default: INFO)')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    
    # Set log level
    logger.remove()  # Remove existing handlers
    logger.add(
        "volcano_deploy_{time}.log",
        rotation="500 MB",
        level=args.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        lambda msg: print(msg, flush=True),
        colorize=True,
        level=args.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    logger.info("Starting Volcano deployment script")
    
    # Get current conda environment
    current_env = get_current_conda_env()
    logger.info(f"Current conda environment: {current_env}")
    
    # Prepare configuration overrides
    config_overrides = {}
    if args.task_name:
        config_overrides['task_name'] = args.task_name
    if args.queue_name:
        config_overrides['queue_name'] = args.queue_name
    if args.conda_env:
        config_overrides['conda_env_name'] = args.conda_env
    if args.image:
        config_overrides['image'] = args.image
    if args.extra_envs:
        default_config = VolcanoConfig()
        config_overrides['extra_envs'] = default_config.extra_envs + args.extra_envs
    
    # Initialize deployment
    deployer = VolcanoDeployment(config_overrides if config_overrides else None)
    
    # Print deployment configuration
    logger.info("\nDeployment configuration summary:")
    logger.info(f"Task command: {args.task_cmd}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"Conda environment: {deployer.config.conda_env_name}")
    logger.info(f"Task name: {deployer.config.task_name}")
    logger.info(f"Queue name: {deployer.config.queue_name}")
    logger.info(f"Image name: {deployer.config.image}")
    if args.extra_envs:
        logger.info(f"Additional environment variables: {args.extra_envs}")
    
    # Confirm deployment
    confirm = input("\nProceed with deployment? [y/N]: ")
    if confirm.lower() != 'y':
        logger.warning("Deployment cancelled by user")
        return
    
    # Execute deployment
    try:
        result = deployer.deploy(args.task_cmd, num_gpus=args.num_gpus, num_replicas=args.num_replicas)
        
        # Print deployment result
        if result.returncode == 0:
            logger.success("Deployment completed successfully")
        else:
            logger.error("Deployment failed")
            
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Errors: {result.stderr}")
            
    except Exception as e:
        logger.exception("Deployment failed with exception")
        raise

if __name__ == "__main__":
    main()
