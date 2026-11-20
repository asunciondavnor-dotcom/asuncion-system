"""Render-only guard, then the existing production Waitress launcher."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def validate(environment):
    if environment.get('PSMS_ENV') != 'production':
        raise RuntimeError('Render requires the explicit production configuration.')
    if environment.get('PSMS_INFERENCE_CACHE_ENABLED') != '0':
        raise RuntimeError('Render requires installed models; runtime model downloads are disabled.')
    if environment.get('VERCEL') not in (None, '', '0'):
        raise RuntimeError('Render must not inherit the Vercel runtime flag.')
    if environment.get('PSMS_PROCUREMENT_RELEASE') or environment.get('PSMS_PROCUREMENT_RELEASE_TOKEN'):
        raise RuntimeError('Render must not inherit a Vercel staging capability.')


if __name__ == '__main__':
    validate(os.environ)
    from serve import main
    main()
