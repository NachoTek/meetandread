"""Hardware detection using psutil for system specifications.

Provides HardwareDetector class to detect system specs including RAM, CPU count,
frequency, and platform information. Used for model size recommendations.
"""

import platform
import time
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class SystemSpecs:
    """System hardware specifications.
    
    Attributes:
        total_ram_gb: Total system RAM in gigabytes
        available_ram_gb: Currently available RAM in gigabytes
        cpu_count_logical: Number of logical CPU cores (includes hyperthreading)
        cpu_count_physical: Number of physical CPU cores
        cpu_freq_mhz: Current CPU frequency in MHz (None if unavailable)
        is_64bit: True if running on 64-bit architecture
        platform: Platform name ('Windows', 'Darwin' for macOS, 'Linux', etc.)
    """
    total_ram_gb: float
    available_ram_gb: float
    cpu_count_logical: int
    cpu_count_physical: int
    cpu_freq_mhz: Optional[float]
    is_64bit: bool
    platform: str


class HardwareDetector:
    """Detects system hardware specifications using psutil.
    
    Provides methods to detect system specs, check minimum requirements,
    and cache results to avoid repeated system calls.
    
    Minimum Requirements:
        - Standard (realtime): 4GB RAM, 2 cores
        - Post-processing (realtime + post-process): 8GB RAM, 4 cores
    
    Example:
        >>> detector = HardwareDetector()
        >>> specs = detector.detect()
        >>> print(f"RAM: {specs.total_ram_gb:.1f}GB, CPUs: {specs.cpu_count_logical}")
        RAM: 16.0GB, CPUs: 8
        >>> detector.has_minimum_requirements(specs)
        True
    """
    
    # Minimum requirements for different modes
    SINGLE_MODE_MIN_RAM_GB = 4.0
    SINGLE_MODE_MIN_CORES = 2
    DUAL_MODE_MIN_RAM_GB = 8.0
    DUAL_MODE_MIN_CORES = 4
    
    def __init__(self, cache_ttl_seconds: int = 60):
        """Initialize the hardware detector.
        
        Args:
            cache_ttl_seconds: How long to cache detection results (default 60)
        """
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cached_specs: Optional[SystemSpecs] = None
        self._cache_timestamp: float = 0
    
    def detect(self) -> SystemSpecs:
        """Detect system hardware specifications.
        
        Returns cached results if within TTL to avoid repeated system calls.
        
        Returns:
            SystemSpecs with detected hardware information
            
        Raises:
            RuntimeError: If detection fails critically
        """
        # Check cache
        if self._cached_specs is not None:
            elapsed = time.time() - self._cache_timestamp
            if elapsed < self._cache_ttl_seconds:
                return self._cached_specs
        
        # Detect RAM
        mem = psutil.virtual_memory()
        total_ram_gb = mem.total / (1024 ** 3)
        available_ram_gb = mem.available / (1024 ** 3)
        
        # Detect CPU
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False) or cpu_count_logical
        
        # Detect CPU frequency (may not be available on all platforms)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else None
        
        # Platform info
        is_64bit = platform.machine().endswith('64')
        platform_name = platform.system()
        
        specs = SystemSpecs(
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            cpu_count_logical=cpu_count_logical,
            cpu_count_physical=cpu_count_physical,
            cpu_freq_mhz=cpu_freq_mhz,
            is_64bit=is_64bit,
            platform=platform_name,
        )
        
        # Cache results
        self._cached_specs = specs
        self._cache_timestamp = time.time()
        
        return specs
    
    def refresh(self) -> SystemSpecs:
        """Force re-detection of hardware, bypassing cache.
        
        Returns:
            Fresh SystemSpecs after re-detection
        """
        self._cached_specs = None
        self._cache_timestamp = 0
        return self.detect()
    
    def has_minimum_requirements(
        self,
        specs: Optional[SystemSpecs] = None,
        dual_mode: bool = False
    ) -> bool:
        """Check if system meets minimum requirements.
        
        Args:
            specs: SystemSpecs to check. If None, calls detect().
            dual_mode: If True, checks post-processing requirements (8GB, 4 cores).
                      If False, checks standard requirements (4GB, 2 cores).
                      
        Returns:
            True if system meets minimum requirements
        """
        if specs is None:
            specs = self.detect()
        
        if dual_mode:
            min_ram = self.DUAL_MODE_MIN_RAM_GB
            min_cores = self.DUAL_MODE_MIN_CORES
        else:
            min_ram = self.SINGLE_MODE_MIN_RAM_GB
            min_cores = self.SINGLE_MODE_MIN_CORES
        
        has_ram = specs.total_ram_gb >= min_ram
        has_cores = specs.cpu_count_logical >= min_cores
        
        return has_ram and has_cores
    
    def get_warning_message(
        self,
        specs: Optional[SystemSpecs] = None,
        dual_mode: bool = False
    ) -> Optional[str]:
        """Get warning message if system is below minimum requirements.
        
        Args:
            specs: SystemSpecs to check. If None, calls detect().
            dual_mode: If True, checks post-processing requirements.
                      
        Returns:
            Warning message string if below minimum, None if requirements met
        """
        if specs is None:
            specs = self.detect()
        
        if self.has_minimum_requirements(specs, dual_mode):
            return None
        
        if dual_mode:
            min_ram = self.DUAL_MODE_MIN_RAM_GB
            min_cores = self.DUAL_MODE_MIN_CORES
            mode_name = "post-processing (realtime + post-process)"
        else:
            min_ram = self.SINGLE_MODE_MIN_RAM_GB
            min_cores = self.SINGLE_MODE_MIN_CORES
            mode_name = "standard (realtime only)"
        
        issues = []
        if specs.total_ram_gb < min_ram:
            issues.append(f"RAM: {specs.total_ram_gb:.1f}GB (need {min_ram}GB+)")
        if specs.cpu_count_logical < min_cores:
            issues.append(f"CPU cores: {specs.cpu_count_logical} (need {min_cores}+")
        
        return f"System may struggle with {mode_name}. Issues: {', '.join(issues)}"
    
    def get_specs_summary(self, specs: Optional[SystemSpecs] = None) -> str:
        """Get a human-readable summary of system specs.
        
        Args:
            specs: SystemSpecs to summarize. If None, calls detect().
            
        Returns:
            Summary string with key specs
        """
        if specs is None:
            specs = self.detect()
        
        freq_str = f"{specs.cpu_freq_mhz:.0f} MHz" if specs.cpu_freq_mhz else "unknown"
        
        return (
            f"{specs.total_ram_gb:.1f}GB RAM, "
            f"{specs.cpu_count_logical} logical cores "
            f"({specs.cpu_count_physical} physical), "
            f"{freq_str}, "
            f"{specs.platform} {'64-bit' if specs.is_64bit else '32-bit'}"
        )

    def get_ram_gb(self, specs: Optional[SystemSpecs] = None) -> float:
        """Get total system RAM in gigabytes.
        
        Args:
            specs: SystemSpecs to use. If None, calls detect().
            
        Returns:
            Total RAM in GB
        """
        if specs is None:
            specs = self.detect()
        return specs.total_ram_gb

    def get_cpu_cores(self, specs: Optional[SystemSpecs] = None) -> int:
        """Get number of logical CPU cores.
        
        Args:
            specs: SystemSpecs to use. If None, calls detect().
            
        Returns:
            Number of logical CPU cores
        """
        if specs is None:
            specs = self.detect()
        return specs.cpu_count_logical

    def get_cpu_frequency(self, specs: Optional[SystemSpecs] = None) -> float:
        """Get CPU frequency in GHz.
        
        Args:
            specs: SystemSpecs to use. If None, calls detect().
            
        Returns:
            CPU frequency in GHz (0.0 if unavailable)
        """
        if specs is None:
            specs = self.detect()
        if specs.cpu_freq_mhz is None:
            return 0.0
        return specs.cpu_freq_mhz / 1000.0  # Convert MHz to GHz
