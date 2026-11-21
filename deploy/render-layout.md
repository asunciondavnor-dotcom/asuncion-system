# Render build settings for the split layout

Apply these settings to the existing new `asuncion-backend` service only after deployment approval.

| Setting | Value |
| --- | --- |
| Git repository | `asunciondavnor-dotcom/asuncion-system` |
| Branch | `main` |
| Root Directory | Repository root (clear the previous `backend/` value) |
| Docker Build Context | Repository root (`.`) |
| Dockerfile Path | `backend/Dockerfile.render` |
| Auto-Deploy | Off |

The repository-root Docker context lets the build assemble Flask's templates and static files from `frontend/`. The runtime manifest still validates the packaged paths. Vercel uses the repository-root `vercel.json` and serves only the lightweight output under `frontend/public`, with application requests rewritten to Render.
