"""Offline image gate. Verify approved bytes and native OCR without importing app."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def deny_network(event, args):
    if event in {'socket.connect', 'socket.connect_ex', 'socket.getaddrinfo'}:
        raise PermissionError('RENDER_IMAGE_NETWORK_FORBIDDEN')
    if event == 'import' and args[0] in {'app', 'db'}:
        raise PermissionError('RENDER_IMAGE_DATABASE_IMPORT_FORBIDDEN')


def main():
    sys.addaudithook(deny_network)
    import inference_releases as releases
    from runtime_configuration import validate_packaged_database_ca
    import os
    releases.require(os.environ.get('PSMS_INFERENCE_CACHE_ENABLED') == '0',
                     'RENDER_RUNTIME_DOWNLOADS_MUST_BE_DISABLED')
    evidence = releases.selected()
    releases.require(evidence['manifest']['release_id'] == 'nlp-online-original-v2'
                     and evidence['files_verified'] == 28, 'RENDER_APPROVED_RELEASE_REQUIRED')
    validate_packaged_database_ca(releases.BASE)
    from PIL import Image, ImageDraw, ImageFont
    import pytesseract
    image = Image.new('RGB', (1000, 160), 'white')
    ImageDraw.Draw(image).text((25, 35), 'SYNTHETIC RENDER 42',
        font=ImageFont.truetype('DejaVuSans.ttf', 45), fill='black')
    text = pytesseract.image_to_string(image, lang='eng', config='--psm 6', timeout=25)
    releases.require('SYNTHETIC' in text and '42' in text, 'RENDER_NATIVE_OCR_FAILED')
    print(json.dumps({'files_verified': 28, 'native_ocr': True,
        'runtime_model_downloads': False, 'database_contacted': False}))


if __name__ == '__main__':
    main()
