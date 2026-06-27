# Re-download NASA GLB models from GitHub
$dest = Join-Path $PSScriptRoot "..\models\nasa"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$downloads = @(
  @{folder='X-57'; file='X-57.glb'; out='x57.glb'},
  @{folder='Space Shuttle (A)'; file='Space Shuttle (A).glb'; out='space-shuttle.glb'},
  @{folder='Ingenuity Mars Helicopter'; file='Ingenuity Mars Helicopter.glb'; out='ingenuity-helicopter.glb'},
  @{folder='Parker Solar Probe'; file='Parker Solar Probe.glb'; out='parker-solar-probe.glb'},
  @{folder='Apollo Lunar Module'; file='Apollo Lunar Module.glb'; out='apollo-lunar-module.glb'}
)

foreach ($d in $downloads) {
  $url = "https://raw.githubusercontent.com/nasa/NASA-3D-Resources/master/3D%20Models/$([uri]::EscapeDataString($d.folder))/$([uri]::EscapeDataString($d.file))"
  $outPath = Join-Path $dest $d.out
  Write-Host "Downloading $($d.out)..."
  Invoke-WebRequest -Uri $url -OutFile $outPath
}

Write-Host "Done. Files in $dest"
