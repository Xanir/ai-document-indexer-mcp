@echo off
REM =============================================================================
REM ai-document-indexer-mcp — Windows convenience launcher
REM
REM Usage:
REM   run.cmd              CPU-only build (fastembed, no GPU)
REM   run.cmd CUDA         NVIDIA GPU build (fastembed-gpu + CUDA runtime)
REM   run.cmd ROCM         AMD GPU build (ROCm runtime + onnxruntime-rocm)
REM =============================================================================
setlocal enabledelayedexpansion

REM ── Parse GPU argument ────────────────────────────────────────────────
set GPU_RUNTIME=%~1
set COMPOSE_FILES=-f docker/compose.yml

if /i "!GPU_RUNTIME!"=="CUDA" (
    set COMPOSE_FILES=!COMPOSE_FILES! -f docker/compose.cuda.yml
    echo === Building with NVIDIA CUDA GPU support ===
) else if /i "!GPU_RUNTIME!"=="ROCM" (
    set COMPOSE_FILES=!COMPOSE_FILES! -f docker/compose.rocm.yml
    echo === Building with AMD ROCm GPU support ===
) else (
    echo === Building CPU-only (no GPU acceleration) ===
)

echo.

echo Stop the Docker container...
docker compose !COMPOSE_FILES! down 2>nul

echo Remove the Docker image...
docker compose !COMPOSE_FILES! rm -f 2>nul

echo Build the Docker image...
docker compose !COMPOSE_FILES! build

echo Run the Docker image...
docker compose !COMPOSE_FILES! up -d

echo.
echo Container is running. Use 'docker logs ai-document-indexer-mcp' to follow output.
