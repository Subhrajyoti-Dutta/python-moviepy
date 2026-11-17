import asyncio
import re
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
STORY_FILE = ROOT / 'assets' / 'story.txt'
INTRO_FILE = ROOT / 'assets' / 'intro.txt'
AUDIO_DIR = ROOT / 'assets' / 'audios'

VOICE = 'en-US-AvaNeural'
RATE = '-5%'
PITCH = '+10Hz'
OVERWRITE = False
MAX_CONCURRENT = 8
RETRIES = 3
RETRY_DELAY_SECONDS = 3


def parse_narration(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing story: {path}')
    story = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    pattern = re.compile(r'^\s*SECTION\s+(\d+)\s*:\s*(.+?)\s*$', re.I | re.M)
    matches = list(pattern.finditer(story))
    if not matches:
        raise ValueError('No SECTION N: headings found in assets/story.txt')

    preface = story[:matches[0].start()].strip().replace("INTRODUCTION", "")
    if preface:
        intro = preface
    elif INTRO_FILE.is_file():
        intro = INTRO_FILE.read_text(encoding='utf-8-sig').strip()
    else:
        raise FileNotFoundError('Put spoken intro before SECTION 1 or in assets/intro.txt')
    if not intro:
        raise ValueError('Introduction is empty')

    paragraphs = []
    for index, match in enumerate(matches):
        if int(match.group(1)) != index + 1:
            raise ValueError('Section numbers must start at 1 and increase consecutively')
        end = matches[index + 1].start() if index + 1 < len(matches) else len(story)
        body = story[match.end():end].strip()
        parts = [re.sub(r'\s+', ' ', part).strip()
                 for part in re.split(r'\n\s*\n', body) if part.strip()]
        if not parts:
            raise ValueError(f'Section {index + 1} has no paragraphs')
        paragraphs.extend(parts)
    return [('intro', re.sub(r'\s+', ' ', intro)),
            *[(f'audio{i}', paragraph) for i, paragraph in enumerate(paragraphs, 1)]]


async def synthesize(name, content, ffmpeg):
    target = AUDIO_DIR / f'{name}.m4a'
    if target.is_file() and not OVERWRITE:
        print(f'Skipped existing: {target.name}', flush=True)
        return False

    temporary_mp3 = AUDIO_DIR / f'.{name}.tmp.mp3'
    temporary_m4a = AUDIO_DIR / f'.{name}.tmp.m4a'
    temporary_mp3.unlink(missing_ok=True)
    temporary_m4a.unlink(missing_ok=True)
    try:
        for attempt in range(1, RETRIES + 1):
            try:
                await edge_tts.Communicate(content, VOICE, rate=RATE, pitch=PITCH).save(str(temporary_mp3))
                if not temporary_mp3.is_file() or temporary_mp3.stat().st_size == 0:
                    raise RuntimeError('Edge TTS returned empty audio')
                break
            except Exception as error:
                temporary_mp3.unlink(missing_ok=True)
                if attempt == RETRIES:
                    raise RuntimeError(f'Failed to synthesize {name} after {RETRIES} attempts') from error
                print(f'Retrying {name} ({attempt}/{RETRIES}): {error}', flush=True)
                await asyncio.sleep(RETRY_DELAY_SECONDS * attempt)

        await asyncio.to_thread(
            subprocess.run,
            [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
             '-i', str(temporary_mp3), '-vn', '-c:a', 'aac', '-b:a', '128k',
             str(temporary_m4a)],
            check=True,
        )
        temporary_m4a.replace(target)
        print(f'Created: {target.name}', flush=True)
        return True
    finally:
        temporary_mp3.unlink(missing_ok=True)
        temporary_m4a.unlink(missing_ok=True)


async def _main():
    entries = parse_narration(STORY_FILE)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    print(f'Voice: {VOICE}; rate: {RATE}; files: {len(entries)}; concurrent tasks: {MAX_CONCURRENT}', flush=True)

    async def worker(name, content):
        async with semaphore:
            try:
                return await synthesize(name, content, ffmpeg), None
            except Exception as error:
                print(f'FAILED: {name}: {error}', flush=True)
                return False, name

    results = await asyncio.gather(*(worker(name, content) for name, content in entries))
    created = sum(was_created for was_created, _ in results)
    failed = [name for _, name in results if name is not None]
    print(f'Finished: {created} created, {len(failed)} failed, {len(entries) - created - len(failed)} skipped.', flush=True)
    if failed:
        raise RuntimeError(f'Audio generation failed for: {", ".join(failed)}. Rerun to retry missing files.')
    print('Done. Run python main.py to render your video.', flush=True)

def main():
    asyncio.run(_main())


if __name__ == '__main__':
    main()