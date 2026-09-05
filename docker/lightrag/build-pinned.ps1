$ErrorActionPreference = "Stop"

$commit = "86e10c4aea9b65f4fcf1d769791d7a3ba4b560d7"
$image = "traffic-rag-lightrag:1.5.7-86e10c4"
$context = "https://github.com/HKUDS/LightRAG.git#$commit"

Write-Host "Building $image from LightRAG commit $commit ..."
docker buildx build `
    --load `
    --tag $image `
    --label "org.opencontainers.image.revision=$commit" `
    $context

if ($LASTEXITCODE -ne 0) {
    throw "LightRAG image build failed with exit code $LASTEXITCODE."
}

$imageId = docker image inspect $image --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) {
    throw "Built image cannot be inspected: $image"
}

$labelsJson = docker image inspect $image --format '{{json .Config.Labels}}'
if ($LASTEXITCODE -ne 0) {
    throw "Built image revision label cannot be inspected: $image"
}
$labels = $labelsJson | ConvertFrom-Json
$revision = $labels.'org.opencontainers.image.revision'

if ($revision -ne $commit) {
    throw "Unexpected image revision '$revision'; expected '$commit'."
}

Write-Host "LightRAG image is ready."
Write-Host "Image:    $image"
Write-Host "Image ID: $($imageId.Trim())"
Write-Host "Revision: $revision"
