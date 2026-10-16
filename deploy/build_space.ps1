# Assembles deploy\hf_space\ (the folder you push to a Hugging Face Docker Space).
param([string[]]$Patients = @('TCGA_CS_4941_19960909','TCGA_CS_4942_19970222','TCGA_CS_4943_20000902','TCGA_CS_5395_19981004','TCGA_CS_5396_20010302'))
$root = Split-Path $PSScriptRoot -Parent
$out  = Join-Path $PSScriptRoot 'hf_space'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory $out, "$out\backend", "$out\outputs", "$out\data" | Out-Null
Copy-Item "$root\backend\app.py" "$out\backend\app.py"
Copy-Item "$root\common.py" $out
Copy-Item "$root\outputs\postprocess_v4.json", "$root\outputs\fold*_model.pth" "$out\outputs"
foreach ($p in $Patients) {
  if (-not (Test-Path "$root\data\$p")) { Write-Warning "missing patient $p"; continue }
  Copy-Item "$root\data\$p" "$out\data\$p" -Recurse
}
Copy-Item "$PSScriptRoot\requirements.txt" $out
Copy-Item "$PSScriptRoot\space_app.py" "$out\app.py"
Copy-Item "$PSScriptRoot\SPACE_README.md" "$out\README.md"
Set-Content "$out\.gitattributes" "*.pth filter=lfs diff=lfs merge=lfs -text`n*.tif filter=lfs diff=lfs merge=lfs -text"
"{0:N0} MB in {1}" -f ((Get-ChildItem $out -Recurse -File | Measure Length -Sum).Sum/1MB), $out
