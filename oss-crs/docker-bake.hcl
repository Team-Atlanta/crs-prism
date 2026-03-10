# =============================================================================
# crs-prism Docker Bake Configuration
# =============================================================================
#
# Builds the CRS base image with Python dependencies for the Prism agent.
#
# Usage:
#   docker buildx bake prepare
#   docker buildx bake --push prepare   # Push to registry
# =============================================================================

variable "REGISTRY" {
  default = "ghcr.io/team-atlanta"
}

variable "VERSION" {
  default = "latest"
}

function "tags" {
  params = [name]
  result = [
    "${REGISTRY}/${name}:${VERSION}",
    "${REGISTRY}/${name}:latest",
    "${name}:latest"
  ]
}

# -----------------------------------------------------------------------------
# Groups
# -----------------------------------------------------------------------------

group "default" {
  targets = ["prepare"]
}

group "prepare" {
  targets = ["prism-base"]
}

# -----------------------------------------------------------------------------
# Base Image
# -----------------------------------------------------------------------------

target "prism-base" {
  context    = "."
  dockerfile = "oss-crs/base.Dockerfile"
  tags       = tags("prism-base")
}
