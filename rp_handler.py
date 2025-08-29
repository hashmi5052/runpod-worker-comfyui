import time
import requests
import json
import base64
import runpod
import os
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from requests.adapters import HTTPAdapter, Retry
from schemas.input import INPUT_SCHEMA


BASE_URI = 'http://127.0.0.1:3000'
VOLUME_MOUNT_PATH = '/runpod-volume'
TIMEOUT = 600

session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
#                               ComfyUI Functions                              #
# ---------------------------------------------------------------------------- #

def wait_for_service(url):
    retries = 0
    while True:
        try:
            requests.get(url)
            return
        except requests.exceptions.RequestException:
            retries += 1
            if retries % 15 == 0:
                logger.info('Service not ready yet. Retrying...')
        except Exception as err:
            logger.error(f'Error: {err}')
        time.sleep(0.2)


def send_get_request(endpoint):
    return session.get(
        url=f'{BASE_URI}/{endpoint}',
        timeout=TIMEOUT
    )


def send_post_request(endpoint, payload):
    return session.post(
        url=f'{BASE_URI}/{endpoint}',
        json=payload,
        timeout=TIMEOUT
    )


def get_filenames(output):
    """Extract image filenames from ComfyUI history output."""
    for key, value in output.items():
        if 'images' in value and isinstance(value['images'], list):
            return value['images']
    return []


def save_init_image(base64_str, filename="init.png"):
    """Save base64 image to ComfyUI input directory."""
    input_dir = f"{VOLUME_MOUNT_PATH}/ComfyUI/input"
    os.makedirs(input_dir, exist_ok=True)
    image_path = os.path.join(input_dir, filename)
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(base64_str))
    return image_path


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #

def handler(event):
    try:
        validated_input = validate(event['input'], INPUT_SCHEMA)

        if 'errors' in validated_input:
            return {'error': validated_input['errors']}

        payload = validated_input['validated_input']

        # Workflow JSON must be provided
        if 'workflow' not in payload:
            return {'error': 'Missing workflow in payload'}

        workflow = payload['workflow']
        logger.info('Received workflow for processing')

        # Handle optional init_image
        if 'init_image' in payload and isinstance(payload['init_image'], str):
            image_path = save_init_image(payload['init_image'])
            # Try to patch the first LoadImage node automatically
            for node_id, node in workflow.items():
                if node.get("class_type", "").lower().startswith("loadimage"):
                    node["inputs"]["image"] = image_path
                    logger.info(f"Patched workflow node {node_id} with uploaded init_image")
                    break

        # Queue workflow to ComfyUI
        logger.debug('Queuing prompt')
        queue_response = send_post_request('prompt', {'prompt': workflow})
        resp_json = queue_response.json()

        if queue_response.status_code != 200:
            logger.error(f'HTTP Status code: {queue_response.status_code}')
            logger.error(json.dumps(resp_json, indent=4, default=str))
            return resp_json

        prompt_id = resp_json.get('prompt_id')
        if not prompt_id:
            return {'error': 'No prompt_id returned by ComfyUI'}

        logger.info(f'Prompt queued successfully: {prompt_id}')

        # Poll until results are ready
        while True:
            r = send_get_request(f'history/{prompt_id}')
            resp_json = r.json()
            if r.status_code == 200 and len(resp_json):
                break
            time.sleep(0.2)

        if not resp_json[prompt_id]['outputs']:
            raise RuntimeError('No output found, check model and workflow validity')

        logger.info(f'Images generated successfully for prompt: {prompt_id}')
        image_filenames = get_filenames(resp_json[prompt_id]['outputs'])

        images = []
        for image_filename in image_filenames:
            filename = image_filename['filename']
            image_path = f'{VOLUME_MOUNT_PATH}/ComfyUI/output/{filename}'
            with open(image_path, 'rb') as image_file:
                images.append(base64.b64encode(image_file.read()).decode('utf-8'))

        return {'images': images}

    except Exception as e:
        logger.error(f'Handler error: {str(e)}')
        return {'error': str(e)}


if __name__ == '__main__':
    wait_for_service(url=f'{BASE_URI}/system_stats')
    logger.info('ComfyUI API is ready')
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start({'handler': handler})
