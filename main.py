from pathlib import Path
import os
import re
import subprocess
import sys
import json
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (AudioFileClip, VideoClip, ImageClip, CompositeVideoClip,
                     CompositeAudioClip)

PROJECT_DIRECTORY = Path(__file__).resolve().parent
ASSETS_DIRECTORY = PROJECT_DIRECTORY / 'assets'
AUDIO_DIRECTORY = ASSETS_DIRECTORY / 'audios'
BACKGROUND_DIRECTORY = ASSETS_DIRECTORY / 'backgrounds'
ICON_DIRECTORY = ASSETS_DIRECTORY / 'icons'
SCRIPT_PATH = ASSETS_DIRECTORY / 'story.txt'
OUTPUT_PATH = PROJECT_DIRECTORY / 'output.mp4'

RESOLUTION = "1080p"
VIDEO_SIZE = (int(RESOLUTION[:-1]) * 16 // 9, int(RESOLUTION[:-1]))
REFERENCE_SIZE = (1920, 1080)
VIDEO_FPS = 30
PARALLEL_RENDER_WORKERS = 8
RENDER_CHUNK_SECONDS = 90
ENCODER_THREADS_PER_WORKER = 1
BACKGROUND_CROSSFADE_SECONDS = 1.0
BACKGROUND_END_TRIM_FRAMES = 3

TRANSITION_IN_SECONDS = 0.50
TRANSITION_OUT_SECONDS = 0.40
TRANSITION_DISTANCE = 75
WORD_STAGGER_SECONDS = 0.06
WORD_ASCENT_SECONDS = 0.375
WORD_EXIT_SECONDS = 0.35
INTRO_PAUSE_SECONDS = 0.5
SECTION_HEADING_SECONDS = 2.0
PARAGRAPH_PAUSE_SECONDS = 0.5  

SUBREDDIT_NAME = 'r/ProRevenge'
AUTHOR_USERNAME = 'u/RedditStories'
STORY_TITLE = open("./assets/story.txt","r").read().split("\n")[0][:-1]

FONT_PATHS = {
    True: ['C:/Windows/Fonts/arialbd.ttf', '/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'],
    False: ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'],
}


def load_font(font_size, bold=False):
    for font_path in FONT_PATHS[bold]:
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, max(1, round(font_size)))
    return ImageFont.load_default()


def load_icon(icon_name, maximum_size):
    icon_path = ICON_DIRECTORY / f'{icon_name}.png'
    if not icon_path.is_file():
        raise FileNotFoundError(f'Missing icon: {icon_path}')
    with Image.open(icon_path) as original:
        icon = original.convert('RGBA')
    bounds = icon.getchannel('A').getbbox()
    if bounds:
        icon = icon.crop(bounds)
    icon.thumbnail((maximum_size, maximum_size), Image.Resampling.LANCZOS)
    return icon


def wrap_text(draw_context, text, text_font, maximum_width):
    lines = []
    current_line = ''
    for word in text.split():
        candidate = f'{current_line} {word}'.strip()
        if draw_context.textbbox((0, 0), candidate, font=text_font)[2] <= maximum_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def create_canvas():
    if abs(VIDEO_SIZE[0] / VIDEO_SIZE[1] - 16 / 9) > 0.01:
        raise ValueError('VIDEO_SIZE must have a 16:9 aspect ratio')
    
    canvas = Image.new('RGBA', VIDEO_SIZE, (0, 0, 0, 0))
    return canvas, ImageDraw.Draw(canvas), VIDEO_SIZE[0] / REFERENCE_SIZE[0]


def render_overlay(draw_function):
    canvas, draw_context, render_scale = create_canvas()
    draw_function(canvas, draw_context, render_scale)
    return np.asarray(canvas)


def create_intro_card(*, background=True, elements=True, title=True):
    def paint(canvas, draw_context, render_scale):
        def scaled(value):
            return round(value * render_scale)

        def draw_text(x, y, text, font_size, *, bold=True):
            draw_context.text((scaled(x), scaled(y)), text, font=load_font(scaled(font_size), bold), fill='#111111', anchor='lt')

        def draw_icon(x, y, icon_name, icon_size):
            icon = load_icon(icon_name, scaled(icon_size))
            canvas.alpha_composite(icon, (scaled(x), scaled(y)))

        def rounded_rectangle(x, y, width, height, radius, *, fill=None, outline=None, outline_width=1):
            draw_context.rounded_rectangle((scaled(x), scaled(y), scaled(x + width), scaled(y + height)), radius=scaled(radius), fill=fill, outline=outline, width=scaled(outline_width))

        title_lines = []
        title_font_size = 57
        for candidate_size in range(96, 54, -3):
            candidate_lines = wrap_text(draw_context, STORY_TITLE, load_font(scaled(candidate_size), True), scaled(1515))
            if len(candidate_lines) <= 3:
                title_lines = candidate_lines
                title_font_size = candidate_size
                break
        if not title_lines:
            raise ValueError('Story title is too long for the introduction card')

        title_line_spacing = title_font_size * 1.22
        last_line_height = draw_context.textbbox((0, 0), title_lines[-1], font=load_font(scaled(title_font_size), True), anchor='lt')[3] / render_scale
        title_height = (len(title_lines) - 1) * title_line_spacing + last_line_height
        buttons_top = 234 + title_height + 51
        card_height = buttons_top + 114 + 66
        card_top = (1080 - card_height) / 2

        if background:
            rounded_rectangle(117, card_top, 1686, card_height, 33, fill='white')
        if not elements:
            return
        draw_icon(201, card_top + 60, 'reddit', 114)
        draw_text(366, card_top + 93, f'{SUBREDDIT_NAME} • {AUTHOR_USERNAME}', 54)
        if title:
            for line_index, line in enumerate(title_lines):
                draw_text(195, card_top + 234 + line_index * title_line_spacing, line, title_font_size)

        button_top = card_top + buttons_top
        for x, width in [(192, 357), (594, 438), (1074, 333)]:
            rounded_rectangle(x, button_top, width, 114, 57, outline='#111111', outline_width=6)
        draw_icon(226.5, button_top + 19.5, 'upvote', 75)
        draw_text(318, button_top + 39, 'Vote', 48)
        draw_icon(448.5, button_top + 19.5, 'downvote', 75)
        draw_icon(636, button_top + 18, 'comment', 78)
        draw_text(735, button_top + 39, 'Comment', 48)
        draw_icon(1113, button_top + 21, 'share', 78)
        draw_text(1215, button_top + 39, 'Share', 48)

    return render_overlay(paint)




STORY_DARKEN_ALPHA = 80  
STORY_TEXT_MARGIN = 123
STORY_HEADING_FONT_SIZE = 75
STORY_PARAGRAPH_FONT_SIZE = 63


def word_layout(text, *, is_heading=False, intro_title=False):
    
    sx = VIDEO_SIZE[0] / REFERENCE_SIZE[0]
    sy = VIDEO_SIZE[1] / REFERENCE_SIZE[1]
    measure = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    def font_for(size):
        return load_font(max(1, round(size * sy)), True)

    def wrap(lines_text, font, max_width):
        return wrap_text(measure, lines_text, font, round(max_width * sx))

    if intro_title:
        lines = []
        font_size = 57
        for candidate in range(96, 54, -3):
            attempt = wrap(STORY_TITLE, font_for(candidate), 1515)
            if len(attempt) <= 3:
                lines, font_size = attempt, candidate
                break
        if not lines:
            raise ValueError('Story title is too long for the introduction card')
        font = font_for(font_size)
        spacing = font_size * 1.22
        last_height = measure.textbbox((0, 0), lines[-1], font=font, anchor='lt')[3] / sy
        title_height = (len(lines) - 1) * spacing + last_height
        buttons_top = 234 + title_height + 51
        card_top = (1080 - (buttons_top + 114 + 66)) / 2
        top, left = card_top + 234, 195
    else:
        display = text.upper() if is_heading else text
        width = 1650 if is_heading else REFERENCE_SIZE[0] - 2 * STORY_TEXT_MARGIN
        font_size = STORY_HEADING_FONT_SIZE if is_heading else STORY_PARAGRAPH_FONT_SIZE
        while font_size >= 33:
            font = font_for(font_size)
            lines = wrap(display, font, width)
            spacing = font_size * (1.25 if is_heading else 1.30)
            heights = [measure.textbbox((0, 0), line, font=font, anchor='lt')[3] / sy for line in lines]
            height = (len(lines) - 1) * spacing + max(heights, default=0)
            if height <= 870:
                break
            font_size -= 3
        else:
            raise ValueError(f'Text is too long to fit: {display[:80]!r}')
        top, left = (1080 - height) / 2, STORY_TEXT_MARGIN

    
    
    
    reference_top = measure.textbbox(
        (0, 0), 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        font=font, anchor='ls'
    )[1]
    words = []
    for line_index, line in enumerate(lines):
        if is_heading and not intro_title:
            x = (VIDEO_SIZE[0] - measure.textlength(line, font=font)) / 2
        else:
            x = left * sx
        line_top = (top + line_index * spacing) * sy
        for word in line.split():
            bbox = measure.textbbox((0, 0), word, font=font, anchor='ls')
            
            
            sprite = Image.new('RGBA',
                               (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])),
                               (0, 0, 0, 0))
            ImageDraw.Draw(sprite).text((-bbox[0], -bbox[1]), word, font=font,
                                        fill='#111111' if intro_title else 'white',
                                        anchor='ls')
            words.append((sprite, round(x + bbox[0]),
                          round(line_top + bbox[1] - reference_top)))
            x += measure.textlength(word + ' ', font=font)
    return words


def add_word_overlay(overlays, text, start_time, duration, *, is_heading=False, intro_title=False):
    
    
    
    
    
    if duration <= 0:
        return
    words = word_layout(text, is_heading=is_heading, intro_title=intro_title)
    if not words:
        return

    count = len(words)
    reveal_budget = max(0.12, duration * 0.525)
    stagger = min(WORD_STAGGER_SECONDS, reveal_budget / max(1, count))
    rise_time = min(WORD_ASCENT_SECONDS, max(0.08, reveal_budget - stagger * (count - 1)))
    entrance_end = (count - 1) * stagger + rise_time
    exit_time = min(WORD_EXIT_SECONDS, duration * 0.25)
    distance = round(TRANSITION_DISTANCE * VIDEO_SIZE[1] / REFERENCE_SIZE[1])

    
    
    
    left = max(0, min(x for sprite, x, y in words))
    top = max(0, min(y for sprite, x, y in words) - distance - 2)
    right = min(VIDEO_SIZE[0], max(x + sprite.width for sprite, x, y in words) + 2)
    bottom = min(VIDEO_SIZE[1], max(y + sprite.height for sprite, x, y in words) + distance + 2)
    local_size = (right - left, bottom - top)
    local_words = [(sprite, x - left, y - top) for sprite, x, y in words]

    full = Image.new('RGBA', local_size, (0, 0, 0, 0))
    for sprite, x, y in local_words:
        full.alpha_composite(sprite, (x, y))
    static_frame = np.asarray(full).copy()
    del full

    from functools import lru_cache

    @lru_cache(maxsize=8)
    def transition_frame(frame_number):
        t = frame_number / VIDEO_FPS
        if entrance_end <= t < duration - exit_time:
            return static_frame
        canvas = Image.new('RGBA', local_size, (0, 0, 0, 0))
        exit_progress = 0.0
        if t >= duration - exit_time:
            exit_progress = ease_in(min(1.0, max(0.0, (t - duration + exit_time) / exit_time)))
        for index, (sprite, x, y) in enumerate(local_words):
            progress = min(1.0, max(0.0, (t - index * stagger) / rise_time))
            if progress <= 0:
                continue
            rise = ease_out(progress)
            alpha = round(255 * rise * (1 - exit_progress))
            if alpha <= 0:
                continue
            offset = round(distance * (1 - rise - exit_progress))
            if alpha == 255:
                canvas.alpha_composite(sprite, (x, y + offset))
            else:
                faded = sprite.copy()
                faded.putalpha(sprite.getchannel('A').point(lambda value: value * alpha // 255))
                canvas.alpha_composite(faded, (x, y + offset))
        return np.asarray(canvas).copy()

    def frame_at(t):
        if entrance_end <= t < duration - exit_time:
            return static_frame
        frame_number = min(max(0, round(t * VIDEO_FPS)), max(0, round(duration * VIDEO_FPS)))
        return transition_frame(frame_number)

    clip = VideoClip(frame_function=lambda t: frame_at(t)[:, :, :3], duration=duration)
    mask = VideoClip(frame_function=lambda t: frame_at(t)[:, :, 3].astype(np.float32) / 255,
                     is_mask=True, duration=duration)
    overlays.append(clip.with_mask(mask).with_position((left, top)).with_start(start_time))


def parse_story(script_path):
    
    if not script_path.is_file():
        raise FileNotFoundError(f'Create {script_path} with SECTION 1: Heading followed by paragraphs')
    script = script_path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    heading_pattern = re.compile(r'^\s*SECTION\s+(\d+)\s*:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)
    matches = list(heading_pattern.finditer(script))
    if not matches:
        raise ValueError('No headings found. Use lines such as SECTION 1: My old company')
    preface = script[:matches[0].start()].strip()
    if preface:
        print('Note: Skipping text before the first SECTION heading (title/introduction).')
    sections = []
    for index, match in enumerate(matches):
        section_number = int(match.group(1))
        if section_number != index + 1:
            raise ValueError('Section numbers must start at 1 and increase consecutively')
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(script)
        body = script[match.end():body_end].strip()
        paragraphs = [paragraph.strip().replace('\n', ' ') for paragraph in re.split(r'\n\s*\n', body) if paragraph.strip()]
        if not paragraphs:
            raise ValueError(f'Section {section_number} has no paragraphs')
        sections.append({'heading': match.group(2).strip(), 'paragraphs': paragraphs})
    return sections


def ease_out(value):
    
    return 1 - (1 - value) ** 3


def ease_in(value):
    
    return value ** 3


def add_overlay(overlays, image, start_time, duration, *, ascent=True, fade_only=False):
    
    
    
    
    
    
    if duration <= 0:
        return

    clip = ImageClip(image, transparent=True).with_duration(duration)
    if not ascent and not fade_only:
        overlays.append(clip.with_start(start_time))
        return

    entrance = min(TRANSITION_IN_SECONDS, duration / 2)
    exit_time = min(TRANSITION_OUT_SECONDS, duration / 2)
    movement = 0 if fade_only else round(TRANSITION_DISTANCE * VIDEO_SIZE[1] / REFERENCE_SIZE[1])

    def animation(time):
        if time < entrance:
            progress = ease_out(max(0.0, time) / entrance)
            return movement * (1 - progress), progress
        if time > duration - exit_time:
            progress = ease_in(min(1.0, (time - (duration - exit_time)) / exit_time))
            return -movement * progress, 1 - progress
        return 0.0, 1.0

    clip = clip.with_position(lambda t: (0, round(animation(t)[0])))
    
    
    if clip.mask is not None:
        animated_mask = clip.mask.transform(
            lambda get_frame, t: get_frame(t) * animation(t)[1]
        )
        clip = clip.with_mask(animated_mask)
    overlays.append(clip.with_start(start_time))


def is_outro_marker(text):
    return bool(re.fullmatch(
        r'\s*(?:SECTION\s+\d+\s*:\s*)?OUTRO\s*[:.!-]?\s*',
        text, flags=re.IGNORECASE
    ))


def audio_duration(path):
    clip = AudioFileClip(str(path))
    try:
        return clip.duration
    finally:
        clip.close()


def create_manifest():
    sections = parse_story(SCRIPT_PATH)
    intro_path = AUDIO_DIRECTORY / 'intro.m4a'
    if not intro_path.is_file():
        raise FileNotFoundError(intro_path)
    intro_duration = audio_duration(intro_path)
    entries = [{'kind': 'intro', 'text': STORY_TITLE, 'start': 0.0,
                'duration': intro_duration, 'audio': str(intro_path)}]
    cursor = intro_duration + INTRO_PAUSE_SECONDS
    paragraph_number = 0
    expected = sum(len(section['paragraphs']) for section in sections)
    for section in sections:
        first_audio = AUDIO_DIRECTORY / f'audio{paragraph_number + 1}.m4a'
        if not first_audio.is_file():
            break
        is_outro = is_outro_marker(section['heading'])
        if not is_outro:
            entries.append({'kind': 'heading', 'text': section['heading'],
                            'start': cursor, 'duration': SECTION_HEADING_SECONDS})
            cursor += SECTION_HEADING_SECONDS
        for paragraph in section['paragraphs']:
            path = AUDIO_DIRECTORY / f'audio{paragraph_number + 1}.m4a'
            if not path.is_file():
                break
            duration = audio_duration(path)
            entries.append({'kind': 'outro' if is_outro or is_outro_marker(paragraph) else 'paragraph',
                            'text': paragraph, 'start': cursor,
                            'duration': duration, 'audio': str(path)})
            cursor += duration
            paragraph_number += 1
            if paragraph_number < expected and (AUDIO_DIRECTORY / f'audio{paragraph_number + 1}.m4a').is_file():
                cursor += PARAGRAPH_PAUSE_SECONDS
    if not paragraph_number:
        raise FileNotFoundError('No audio1.m4a found')
    if paragraph_number < expected:
        print(f'WARNING: only {paragraph_number}/{expected} paragraph audio files exist.', flush=True)
    return {'entries': entries, 'duration': cursor,
            'fps': VIDEO_FPS, 'size': list(VIDEO_SIZE)}


def ffmpeg_source_duration(path, ffmpeg):
    result = subprocess.run([ffmpeg, '-hide_banner', '-i', str(path)],
                            capture_output=True, text=True)
    match = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', result.stderr)
    if not match:
        raise RuntimeError(f'Cannot determine background duration: {path}\n{result.stderr[-1000:]}')
    return int(match[1])*3600 + int(match[2])*60 + float(match[3])


def prepare_ffmpeg_background():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    sources = [BACKGROUND_DIRECTORY / f'clip_{i}.mp4' for i in range(1, 7)]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = PROJECT_DIRECTORY / '.moviepy_cache'
    cache.mkdir(exist_ok=True)
    signature = json.dumps({'sources': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sources],
                            'size': VIDEO_SIZE, 'fps': VIDEO_FPS, 'crossfade': BACKGROUND_CROSSFADE_SECONDS,
                            'trim': BACKGROUND_END_TRIM_FRAMES, 'tint': STORY_DARKEN_ALPHA}, sort_keys=True)
    digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
    target = cache / f'background_{digest}.mp4'
    if target.is_file() and target.stat().st_size > 1000:
        print('Using cached FFmpeg background:', target.name, flush=True)
        return target
    durations = [ffmpeg_source_duration(p, ffmpeg) - BACKGROUND_END_TRIM_FRAMES / VIDEO_FPS for p in sources]
    transition = BACKGROUND_CROSSFADE_SECONDS
    if any(d <= transition for d in durations):
        raise ValueError('Background shorter than crossfade')
    width, height = VIDEO_SIZE
    filters = []
    for i, duration in enumerate(durations):
        filters.append(f'[{i}:v:0]trim=duration={duration:.6f},setpts=PTS-STARTPTS,'
                       f'scale={width}:{height}:force_original_aspect_ratio=increase,'
                       f'crop={width}:{height},fps={VIDEO_FPS},format=yuv420p[v{i}]')
    prev = 'v0'
    cumulative = durations[0]
    for i in range(1, len(sources)):
        offset = cumulative - transition
        filters.append(f'[{prev}][v{i}]xfade=transition=fade:duration={transition}:offset={offset:.6f}[x{i}]')
        prev = f'x{i}'
        cumulative += durations[i] - transition
    brightness = 1 - STORY_DARKEN_ALPHA / 255
    filters.append(f'[{prev}]colorchannelmixer=rr={brightness:.6f}:gg={brightness:.6f}:bb={brightness:.6f},format=yuv420p[final]')
    temporary = cache / f'background_{digest}.working.mp4'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y']
    for source in sources:
        command += ['-i', str(source)]
    command += ['-filter_complex', ';'.join(filters), '-map', '[final]', '-an',
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-threads', '2',
                '-pix_fmt', 'yuv420p', str(temporary)]
    print('Preprocessing FFmpeg background once...', flush=True)
    try:
        subprocess.run(command, check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def render_video_segment(start_frame, end_frame, output_path, manifest_path, background_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    duration = manifest['duration']
    segment_start = start_frame / VIDEO_FPS
    segment_end = min(duration, end_frame / VIDEO_FPS)
    segment_duration = segment_end - segment_start
    if segment_duration <= 0:
        raise ValueError('Empty segment')
    overlays = []
    audio_layers = []
    opened = []
    foreground = None
    mixed_audio = None
    audio_path = Path(output_path).with_suffix('.audio.wav')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        for entry in manifest['entries']:
            entry_start = entry['start']
            entry_end = entry_start + entry['duration']
            if entry_end <= segment_start or entry_start >= segment_end:
                continue
            kind = entry['kind']
            local_start = entry_start - segment_start
            if kind == 'intro':
                add_overlay(overlays, create_intro_card(background=True, elements=False),
                            local_start, entry['duration'], ascent=False, fade_only=True)
                add_overlay(overlays, create_intro_card(background=False, elements=True, title=False),
                            local_start, entry['duration'])
                add_word_overlay(overlays, STORY_TITLE, local_start, entry['duration'], intro_title=True)
            elif kind == 'heading':
                add_word_overlay(overlays, entry['text'], local_start,
                                 entry['duration'], is_heading=True)
            elif kind == 'paragraph':
                add_word_overlay(overlays, entry['text'], local_start, entry['duration'])
            if 'audio' in entry:
                clip = AudioFileClip(entry['audio'])
                opened.append(clip)
                trim_start = max(0.0, segment_start - entry_start)
                trim_end = min(clip.duration, segment_end - entry_start)
                if trim_end > trim_start:
                    audio_layers.append(clip.subclipped(trim_start, trim_end).with_start(max(0, local_start)))
        mixed_audio = CompositeAudioClip(audio_layers).with_duration(segment_duration)
        mixed_audio.write_audiofile(str(audio_path), fps=24000, nbytes=2,
                                    codec='pcm_s16le', logger=None)
        foreground = (CompositeVideoClip(overlays, size=VIDEO_SIZE, bg_color=None)
                      if overlays else ImageClip(np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 4), dtype=np.uint8), transparent=True))
        foreground = foreground.with_duration(segment_duration)
        width, height = VIDEO_SIZE
        frame_count = end_frame - start_frame
        command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                   '-stream_loop', '-1', '-ss', f'{segment_start:.6f}', '-i', str(background_path),
                   '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{width}x{height}',
                   '-r', str(VIDEO_FPS), '-i', 'pipe:0', '-i', str(audio_path),
                   '-filter_complex', '[0:v][1:v]overlay=shortest=1:format=auto,format=yuv420p[v]',
                   '-map', '[v]', '-map', '2:a:0', '-frames:v', str(frame_count),
                   '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                   '-threads', str(ENCODER_THREADS_PER_WORKER),
                   '-c:a', 'aac', '-b:a', '192k', '-shortest', str(output_path)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame_index in range(frame_count):
                t = frame_index / VIDEO_FPS
                rgb = foreground.get_frame(t)
                alpha = (np.clip(foreground.mask.get_frame(t), 0, 1) * 255).astype(np.uint8)
                rgba = np.empty((height, width, 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb[:, :, :3]
                rgba[:, :, 3] = alpha
                process.stdin.write(rgba.tobytes())
            process.stdin.close()
            error = process.stderr.read().decode(errors='replace')
            if process.wait() != 0:
                raise RuntimeError(f'FFmpeg segment failed: {error[-3000:]}')
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        finally:
            process.stderr.close()
    finally:
        if foreground is not None:
            foreground.close()
        if mixed_audio is not None:
            mixed_audio.close()
        for clip in opened:
            clip.close()
        audio_path.unlink(missing_ok=True)


def main():
    manifest = create_manifest()
    background = prepare_ffmpeg_background()
    total_frames = math.ceil(manifest['duration'] * VIDEO_FPS)
    chunk_frames = max(VIDEO_FPS, round(RENDER_CHUNK_SECONDS * VIDEO_FPS))
    ranges = [(i, min(i + chunk_frames, total_frames)) for i in range(0, total_frames, chunk_frames)]
    temp_dir = PROJECT_DIRECTORY / '.render_segments'
    temp_dir.mkdir(exist_ok=True)
    manifest_path = temp_dir / 'timeline.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    paths = [temp_dir / f'part_{i:04d}.mp4' for i in range(len(ranges))]
    print(f'Rendering {len(paths)} segments with {PARALLEL_RENDER_WORKERS} workers.', flush=True)

    def run_worker(index):
        a, b = ranges[index]
        command = [sys.executable, str(Path(__file__).resolve()), '--render-segment',
                   str(a), str(b), str(paths[index]), str(manifest_path), str(background)]
        log_path = temp_dir / f'part_{index:04d}.log'
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=str(PROJECT_DIRECTORY), stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'Segment {index} failed; see {log_path}')
        print(f'Completed segment {index + 1}/{len(paths)}', flush=True)

    with ThreadPoolExecutor(max_workers=max(1, PARALLEL_RENDER_WORKERS)) as pool:
        futures = [pool.submit(run_worker, i) for i in range(len(paths))]
        for future in as_completed(futures):
            future.result()
    concat_list = temp_dir / 'concat.txt'
    concat_list.write_text(''.join(f"file '{p.name}'\n" for p in paths), encoding='utf-8')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'concat', '-safe', '0', '-i', str(concat_list),
                    '-c', 'copy', str(OUTPUT_PATH)], check=True)
    print(f'Finished: {OUTPUT_PATH}', flush=True)
    print(f'Temporary segments preserved at: {temp_dir}', flush=True)


if __name__ == '__main__':
    import generate_audio_parallel as gap
    import shutil
    shutil.rmtree("assets\\audios")
    gap.main()
    if len(sys.argv) == 7 and sys.argv[1] == '--render-segment':
        render_video_segment(int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]),
                             Path(sys.argv[5]), Path(sys.argv[6]))
    else:
        main()
