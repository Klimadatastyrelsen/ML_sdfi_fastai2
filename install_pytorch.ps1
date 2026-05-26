# install_pytorch.ps1 — install PyTorch with CUDA for Windows.
# Run after: conda activate ML_sdfi
# Override: $env:PYTORCH_CUDA = "cu124" | "cu121" | "cu118" | "cu128-nightly"

$ErrorActionPreference = "Stop"

function Get-PytorchVariant {
    if ($env:PYTORCH_CUDA) { return $env:PYTORCH_CUDA }
    if ($env:INSTALL_PYTORCH_NO_GPU -eq "1") { return "cu124" }

    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvsmi) {
        Write-Error "nvidia-smi not found; CUDA GPU required for ML_sdfi environment"
    }
    $name = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    $cap = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1) -replace '\s', ''
    if ($cap -eq "12.0" -or $name -match 'Blackwell|RTX 50') { return "cu128-nightly" }
    return "cu124"
}

function Install-PytorchVariant {
    param([string]$Variant, [string[]]$Force)
    switch ($Variant) {
        "cu128-nightly" {
            pip install @Force --pre torch torchvision torchaudio `
                --index-url https://download.pytorch.org/whl/nightly/cu128
        }
        "cu124" {
            pip install @Force torch torchvision torchaudio `
                --index-url https://download.pytorch.org/whl/cu124
        }
        "cu121" {
            pip install @Force torch torchvision torchaudio `
                --index-url https://download.pytorch.org/whl/cu121
        }
        "cu118" {
            pip install @Force torch torchvision torchaudio `
                --index-url https://download.pytorch.org/whl/cu118
        }
        default { throw "Unknown PYTORCH_CUDA=$Variant (use cu124, cu121, cu118, or cu128-nightly)" }
    }
}

function Test-CudaAvailable {
    $out = python -c @"
import sys, torch
ok = torch.cuda.is_available()
print(f'PYTORCH_INSTALL: cuda_available={ok}')
print(f'PYTORCH_INSTALL: torch={torch.__version__} cuda={torch.version.cuda}')
if ok:
    print(f'PYTORCH_INSTALL: device={torch.cuda.get_device_name(0)}')
elif __import__('os').environ.get('INSTALL_PYTORCH_NO_GPU') == '1':
    print('PYTORCH_INSTALL: skipping CUDA smoke test (INSTALL_PYTORCH_NO_GPU=1)')
else:
    print('CUDA_UNAVAILABLE: torch.cuda.is_available() is False after install', file=sys.stderr)
    sys.exit(1)
"@ 2>&1
    $out | ForEach-Object { Write-Host $_ }
    return $LASTEXITCODE -eq 0
}

$variant = Get-PytorchVariant
Write-Host "PYTORCH_INSTALL: selected=$variant"

$force = @()
try {
    $current = python -c "import torch; v=getattr(torch.version,'cuda',None); print('none' if not v else 'cu128-nightly' if '12.8' in str(v) or str(v).startswith('13.') else 'cu124')" 2>$null
} catch { $current = "none" }
if ($current -and $current -ne $variant -and $current -ne "none") {
    Write-Host "Replacing PyTorch ($current -> $variant)"
    $force = @("--force-reinstall")
}

Install-PytorchVariant -Variant $variant -Force $force
if (Test-CudaAvailable) { exit 0 }

# Older NVIDIA drivers (e.g. 496.x) cannot run cu124; retry cu118.
if ($variant -eq "cu124" -and -not $env:PYTORCH_CUDA) {
    Write-Host "PYTORCH_INSTALL: cu124 CUDA check failed; retrying with cu118 (older driver)"
    Install-PytorchVariant -Variant "cu118" -Force @("--force-reinstall")
    if (Test-CudaAvailable) { exit 0 }
}

exit 1
