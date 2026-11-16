"""Reddit story video builder (MoviePy 2.x).

Install: pip install moviepy pillow numpy

Script format in assets/story.txt:
    SECTION 1: My old company

    First paragraph of this section.

    Second paragraph of this section.

    SECTION 2: The problem

    First paragraph of section two.

One narration file per PARAGRAPH in reading order across all sections:
assets/audios/audio1.m4a, audio2.m4a, etc. Each paragraph is displayed
for the exact duration of its corresponding narration file.
"""
from pathlib import Path
import re
import subprocess
import sys
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (AudioFileClip, VideoFileClip, VideoClip, ImageClip, ColorClip, CompositeVideoClip,
                     CompositeAudioClip, concatenate_videoclips)
from moviepy.video.fx import Loop

PROJECT_DIRECTORY = Path(__file__).resolve().parent
ASSETS_DIRECTORY = PROJECT_DIRECTORY / 'assets'
AUDIO_DIRECTORY = ASSETS_DIRECTORY / 'audios'
BACKGROUND_DIRECTORY = ASSETS_DIRECTORY / 'backgrounds'
ICON_DIRECTORY = ASSETS_DIRECTORY / 'icons'
SCRIPT_PATH = ASSETS_DIRECTORY / 'story.txt'
OUTPUT_PATH = PROJECT_DIRECTORY / 'output.mp4'

RESOLUTION = "240p"
VIDEO_SIZE = (int(RESOLUTION[:-1]) * 16 // 9, int(RESOLUTION[:-1]))
REFERENCE_SIZE = (1920, 1080)
VIDEO_FPS = 30
# Independent FFmpeg/Python worker processes; 2 is safer on low-RAM laptops.
PARALLEL_RENDER_WORKERS = 8
# Chunk length in seconds; all cuts are aligned to output frames.
RENDER_CHUNK_SECONDS = 90
# Each worker runs its own encoder. Too many threads per worker can slow rendering.
ENCODER_THREADS_PER_WORKER = 2
BACKGROUND_CROSSFADE_SECONDS = 1.0
# Avoid MoviePy requesting the nonexistent end-boundary frame of source videos.
BACKGROUND_END_TRIM_FRAMES = 3

# Vertical ascent transitions (1920 x 1080 reference pixels).
TRANSITION_IN_SECONDS = 0.50
TRANSITION_OUT_SECONDS = 0.40
TRANSITION_DISTANCE = 75
WORD_STAGGER_SECONDS = 0.06
WORD_ASCENT_SECONDS = 0.375
WORD_EXIT_SECONDS = 0.35
INTRO_PAUSE_SECONDS = 1.0
SECTION_HEADING_SECONDS = 2.0
PARAGRAPH_PAUSE_SECONDS = 1.0  # Silence between paragraphs and before the next section

SUBREDDIT_NAME = 'r/ProRevenge'
AUTHOR_USERNAME = 'u/RedditStories'
STORY_TITLE = 'My Coworker Got Me Fired Over $5,000. Years Later, I Got My Revenge.'

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
    # Draw at native output resolution, never supersample every overlay.
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


# Clipchamp-style narration overlays (reference coordinates: 1920 x 1080).
# The Reddit-style white card is used ONLY for the introduction.
STORY_DARKEN_ALPHA = 80  # Applied continuously from the first frame through the last.
STORY_TEXT_MARGIN = 123
STORY_HEADING_FONT_SIZE = 75
STORY_PARAGRAPH_FONT_SIZE = 63


def create_story_card(text, *, is_heading=False):
    """White Arial text directly on subtly darkened footage; no card or border.

    Headings are uppercase and center-aligned. Paragraphs are bold,
    left-aligned within a horizontally centered block. Both text blocks
    are vertically centered, matching the supplied Clipchamp screenshots.
    """
    def paint(canvas, draw_context, render_scale):
        def scale(value):
            return round(value * render_scale)

        # A full-frame transparent tint improves readability while allowing
        # the background footage to remain visible and continuously moving.
        draw_context.rectangle(
            (0, 0, canvas.width, canvas.height),
            fill=(0, 0, 0, STORY_DARKEN_ALPHA),
        )

        if not text.strip():
            return
        display_text = text.upper() if is_heading else text
        max_width = 1650 if is_heading else REFERENCE_SIZE[0] - 2 * STORY_TEXT_MARGIN
        font_size = STORY_HEADING_FONT_SIZE if is_heading else STORY_PARAGRAPH_FONT_SIZE
        max_height = 870

        while font_size >= 33:
            font = load_font(scale(font_size), bold=True)
            lines = wrap_text(draw_context, display_text, font, scale(max_width))
            line_spacing = font_size * (1.25 if is_heading else 1.30)
            # Measure visible glyph bounds to center the complete text block.
            line_heights = [
                draw_context.textbbox((0, 0), line, font=font, anchor='lt')[3]
                / render_scale for line in lines
            ]
            text_height = (len(lines) - 1) * line_spacing + max(line_heights, default=0)
            if text_height <= max_height:
                break
            font_size -= 3
        else:
            raise ValueError(f'Text is too long to fit on screen: {display_text[:80]!r}')

        text_top = (REFERENCE_SIZE[1] - text_height) / 2
        for index, line in enumerate(lines):
            if is_heading:
                line_width = draw_context.textbbox((0, 0), line, font=font)[2] / render_scale
                text_left = (REFERENCE_SIZE[0] - line_width) / 2
            else:
                text_left = STORY_TEXT_MARGIN
            draw_context.text(
                (scale(text_left), scale(text_top + index * line_spacing)),
                line, font=font, fill='white', anchor='lt',
            )

    return render_overlay(paint)


def word_layout(text, *, is_heading=False, intro_title=False):
    """Lay out word sprites ONCE at output resolution (no 3x supersampling)."""
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

    # All words on a line must share ONE font baseline. Using anchor='lt'
    # independently for each word makes words with different glyph heights
    # (e.g. "any", "A", "company") jump vertically.
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
            # Draw the glyph into a tight sprite while retaining its offset
            # from the common baseline, including descenders and ascenders.
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
    """Staggered words, rendered only during transitions; static middle is reused.

    No full-resolution supersampling, no LANCZOS resize on each frame, and
    no Pillow work while text is stationary. RGB and alpha share a cache.
    """
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

    # Crop the animated overlay to its actual text bounds. Previously every
    # paragraph allocated/composited a full 1920x1080 RGBA frame on every
    # transition frame, even though most pixels were transparent.
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
    """Requires explicit SECTION N: Heading lines and blank-line-separated paragraphs."""
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


def open_background_video(video_path):
    """Open a background clip, removing unsupported data tracks if necessary.

    The original video is never changed. A video-only MP4 is cached beside the
    project in .moviepy_cache when MoviePy cannot parse the original metadata.
    """
    try:
        return VideoFileClip(str(video_path), audio=False)
    except (OSError, ValueError) as original_error:
        print(f"WARNING: MoviePy could not read {video_path.name}: {original_error.__class__.__name__}")
        print("Attempting a video-only remux to remove timecode/data streams...")

    cache_directory = PROJECT_DIRECTORY / '.moviepy_cache'
    cache_directory.mkdir(exist_ok=True)
    cleaned_path = cache_directory / f'{video_path.stem}_video_only.mp4'
    source_stat = video_path.stat()
    needs_remux = (not cleaned_path.exists() or
                   cleaned_path.stat().st_mtime_ns < source_stat.st_mtime_ns)

    if needs_remux:
        temporary_path = cache_directory / f'{video_path.stem}_working.mp4'
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
            '-y', '-i', str(video_path), '-map', '0:v:0', '-an', '-sn', '-dn',
            '-c:v', 'copy', '-map_metadata', '-1', '-movflags', '+faststart',
            str(temporary_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            temporary_path.replace(cleaned_path)
        except subprocess.CalledProcessError as error:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f'Unable to clean {video_path.name}. FFmpeg reported:\n{error.stderr}'
            ) from error

    try:
        cleaned_clip = VideoFileClip(str(cleaned_path), audio=False)
    except Exception as error:
        raise RuntimeError(
            f'MoviePy still cannot open the cleaned copy of {video_path.name}: '
            f'{cleaned_path}'
        ) from error
    print(f'Using cleaned background: {cleaned_path.name}')
    return cleaned_clip


def fit_background(source_clip):
    # MoviePy/FFmpeg can report a duration that reaches one frame beyond the
    # last decodable frame. Trim three source frames before resizing/looping.
    # This does not re-encode the file or add a preprocessing pass.
    source_fps = source_clip.fps or VIDEO_FPS
    safe_end = source_clip.duration - BACKGROUND_END_TRIM_FRAMES / source_fps
    if safe_end <= BACKGROUND_CROSSFADE_SECONDS:
        raise ValueError('Background clip is too short after end-frame safety trim')
    source_clip = source_clip.subclipped(0, safe_end)
    scale_factor = max(VIDEO_SIZE[0] / source_clip.w, VIDEO_SIZE[1] / source_clip.h)
    resized_clip = source_clip.resized(scale_factor)
    return resized_clip.cropped(x_center=resized_clip.w / 2, y_center=resized_clip.h / 2, width=VIDEO_SIZE[0], height=VIDEO_SIZE[1]).without_audio()


def build_background(background_sources, duration):
    """Cycle through all six clips with crossfades and loop the resulting sequence."""
    transition = BACKGROUND_CROSSFADE_SECONDS
    prepared = [fit_background(source) for source in background_sources]
    if any(clip.duration <= transition for clip in prepared):
        raise ValueError('Every background clip must be longer than the crossfade duration')
    sequence = []
    for index, clip in enumerate(prepared):
        # Fade in each clip except the first; overlapping clips create a crossfade.
        if index:
            from moviepy.video.fx import CrossFadeIn
            clip = clip.with_effects([CrossFadeIn(transition)])
        sequence.append(clip)
    # Composite the clips at overlapping start times.
    layers = []
    position = 0.0
    for clip in sequence:
        layers.append(clip.with_start(position))
        position += clip.duration - transition
    cycle_duration = position + transition
    cycle = CompositeVideoClip(layers, size=VIDEO_SIZE).with_duration(cycle_duration)
    # A loop boundary is a hard cut; crossfades occur between successive source clips.
    return cycle.with_effects([Loop(duration=duration)]).with_duration(duration)


def ease_out(value):
    """Cubic ease-out, for the upward entrance."""
    return 1 - (1 - value) ** 3


def ease_in(value):
    """Cubic ease-in, for the upward exit."""
    return value ** 3


def add_overlay(overlays, image, start_time, duration, *, ascent=True, fade_only=False):
    """Animate a transparent overlay without moving or fading the footage.

    Entrance: rise from below while fading in. Exit: rise above while
    fading out. Image position and alpha are evaluated at video time, so
    no extra frames are rendered or encoded in advance.
    """
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
    # Fade the existing transparency mask, not the RGB image. This preserves
    # white text and the card's true white fill throughout the animation.
    if clip.mask is not None:
        animated_mask = clip.mask.transform(
            lambda get_frame, t: get_frame(t) * animation(t)[1]
        )
        clip = clip.with_mask(animated_mask)
    overlays.append(clip.with_start(start_time))


def build_timeline(sections, intro_narration, paragraph_narrations):
    """Match each paragraph 1:1 with its narration, across section boundaries."""
    overlays = []
    audio_layers = [intro_narration.with_start(0)]
    current_time = intro_narration.duration
    add_overlay(overlays, create_intro_card(background=True, elements=False),
                0, intro_narration.duration, ascent=False, fade_only=True)
    add_overlay(overlays, create_intro_card(background=False, elements=True, title=False),
                0, intro_narration.duration)
    add_word_overlay(overlays, STORY_TITLE, 0, intro_narration.duration, intro_title=True)
    current_time += INTRO_PAUSE_SECONDS

    narration_index = 0
    total_paragraphs = sum(len(s['paragraphs']) for s in sections)
    for section in sections:
        # A heading is displayed only if at least one paragraph in this
        # section has an audio file available.
        if narration_index >= len(paragraph_narrations):
            break
        # The OUTRO is audio-only: do not show its heading or reserve
        # heading-screen time. The existing paragraph pause is preserved.
        is_outro_section = section['heading'].strip().casefold() == 'outro'
        if not is_outro_section:
            add_word_overlay(overlays, section['heading'], current_time,
                             SECTION_HEADING_SECONDS, is_heading=True)
            current_time += SECTION_HEADING_SECONDS

        for paragraph in section['paragraphs']:
            if narration_index >= len(paragraph_narrations):
                break
            narration = paragraph_narrations[narration_index]
            narration_index += 1
            paragraph_duration = narration.duration
            # Both the visible paragraph and its audio start at precisely
            # the same time and end after the audio's actual duration.
            audio_layers.append(narration.with_start(current_time))
            # The final paragraph is the spoken outro. Keep its audio,
            # but leave the screen free for a separate outro page.
            is_outro = narration_index == total_paragraphs
            if not is_outro:
                add_word_overlay(overlays, paragraph, current_time,
                                 paragraph_duration)
            current_time += paragraph_duration
            # No trailing silence after the final narration.
            if narration_index < len(paragraph_narrations):
                current_time += PARAGRAPH_PAUSE_SECONDS

    return overlays, CompositeAudioClip(audio_layers), current_time


def render_video_segment(start_frame=None, end_frame=None, output_path=None):
    sections = parse_story(SCRIPT_PATH)
    # Audio numbering follows PARAGRAPHS, not sections. Example: if section 1
    # has three paragraphs, audio1/audio2/audio3 belong to section 1;
    # section 2 begins at audio4. Never estimate durations from word counts.
    paragraph_count = sum(len(section['paragraphs']) for section in sections)
    narration_paths = []
    for paragraph_number in range(1, paragraph_count + 1):
        narration_path = AUDIO_DIRECTORY / f'audio{paragraph_number}.m4a'
        if not narration_path.is_file():
            break
        narration_paths.append(narration_path)

    if not narration_paths:
        raise FileNotFoundError(
            f'No paragraph narration found. Expected {AUDIO_DIRECTORY / "audio1.m4a"}'
        )
    if len(narration_paths) < paragraph_count:
        print(
            f'WARNING: story.txt has {paragraph_count} paragraphs, but only '
            f'{len(narration_paths)} consecutive paragraph audio files were found. '
            f'Rendering the first {len(narration_paths)} paragraphs only. '
            'Remaining paragraphs and their section headings will be omitted.'
        )
    else:
        print(f'Found {paragraph_count} paragraph audio files: 1:1 timing enabled.')

    required_paths = [AUDIO_DIRECTORY / 'intro.m4a', *narration_paths, *(BACKGROUND_DIRECTORY / f'clip_{index}.mp4' for index in range(1, 7))]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required asset: {path}')

    opened_clips = []
    final_video = None
    try:
        intro_narration = AudioFileClip(str(required_paths[0]))
        opened_clips.append(intro_narration)
        paragraph_narrations = []
        for path in narration_paths:
            narration = AudioFileClip(str(path))
            opened_clips.append(narration)
            paragraph_narrations.append(narration)
        background_sources = []
        for index in range(1, 7):
            clip = open_background_video(BACKGROUND_DIRECTORY / f'clip_{index}.mp4')
            opened_clips.append(clip)
            background_sources.append(clip)

        overlays, audio, total_duration = build_timeline(sections, intro_narration, paragraph_narrations)
        background = build_background(background_sources, total_duration)
        # A single constant-opacity black layer covers the entire video, including
        # the intro, pauses, headings, and paragraphs. No sudden tint changes.
        continuous_tint = ColorClip(size=VIDEO_SIZE, color=(0, 0, 0),
                                    duration=total_duration).with_opacity(STORY_DARKEN_ALPHA / 255)
        final_video = CompositeVideoClip([background, continuous_tint, *overlays],
                                         size=VIDEO_SIZE).with_audio(audio)
        if start_frame is not None:
            # Frame-aligned boundaries prevent gaps or duplicated video frames.
            start_time = start_frame / VIDEO_FPS
            end_time = min(total_duration, end_frame / VIDEO_FPS)
            if start_time >= total_duration:
                raise ValueError('Segment begins after the end of the video')
            segment = final_video.subclipped(start_time, end_time)
        else:
            segment = final_video
        try:
            segment.write_videofile(
                str(output_path or OUTPUT_PATH), fps=VIDEO_FPS, codec='libx264',
                audio_codec='aac', preset='ultrafast', threads=ENCODER_THREADS_PER_WORKER,
                ffmpeg_params=['-crf', '18'], logger='bar',
            )
        finally:
            if segment is not final_video:
                segment.close()
    finally:
        if final_video is not None:
            final_video.close()
        for clip in opened_clips:
            clip.close()


def get_total_duration():
    """Inspect audio lengths without decoding video or creating any overlays."""
    sections = parse_story(SCRIPT_PATH)
    count = sum(len(section['paragraphs']) for section in sections)
    clips = []
    try:
        intro = AudioFileClip(str(AUDIO_DIRECTORY / 'intro.m4a'))
        clips.append(intro)
        total = intro.duration + INTRO_PAUSE_SECONDS
        n = 0
        for section in sections:
            if n >= count or not (AUDIO_DIRECTORY / f'audio{n+1}.m4a').exists():
                break
            if section['heading'].strip().casefold() != 'outro':
                total += SECTION_HEADING_SECONDS
            for _ in section['paragraphs']:
                path = AUDIO_DIRECTORY / f'audio{n+1}.m4a'
                if not path.exists():
                    return total
                clip = AudioFileClip(str(path))
                clips.append(clip)
                total += clip.duration
                n += 1
                if n < count and (AUDIO_DIRECTORY / f'audio{n+1}.m4a').is_file():
                    total += PARAGRAPH_PAUSE_SECONDS
        return total
    finally:
        for clip in clips:
            clip.close()


def main():
    if PARALLEL_RENDER_WORKERS <= 1:
        render_video_segment()
        return
    duration = get_total_duration()
    total_frames = int(np.ceil(duration * VIDEO_FPS))
    chunk_frames = max(VIDEO_FPS, int(RENDER_CHUNK_SECONDS * VIDEO_FPS))
    ranges = [(i, min(i + chunk_frames, total_frames))
              for i in range(0, total_frames, chunk_frames)]
    if len(ranges) == 1:
        render_video_segment()
        return
    temp_dir = PROJECT_DIRECTORY / '.render_segments'
    temp_dir.mkdir(exist_ok=True)
    paths = [temp_dir / f'part_{i:04d}.mp4' for i in range(len(ranges))]
    print(f'Rendering {len(paths)} segments with {PARALLEL_RENDER_WORKERS} workers.', flush=True)

    def run_worker(index):
        a, b = ranges[index]
        command = [sys.executable, str(Path(__file__).resolve()),
                   '--render-segment', str(a), str(b), str(paths[index])]
        # Avoid multiple progress bars fighting for the same terminal.
        with (temp_dir / f'part_{index:04d}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=str(PROJECT_DIRECTORY), stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'Segment {index} failed; see {temp_dir / f"part_{index:04d}.log"}')
        print(f'Completed segment {index + 1}/{len(paths)}', flush=True)

    with ThreadPoolExecutor(max_workers=PARALLEL_RENDER_WORKERS) as pool:
        futures = [pool.submit(run_worker, i) for i in range(len(paths))]
        for future in as_completed(futures):
            future.result()

    # Stream-copy concat: no additional video encoding pass.
    concat_list = temp_dir / 'concat.txt'
    concat_list.write_text(''.join(f"file '{p.name}'\n" for p in paths), encoding='utf-8')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    '-f', 'concat', '-safe', '0', '-i', str(concat_list),
                    '-c', 'copy', str(OUTPUT_PATH)], check=True)

    print(f'Finished: {OUTPUT_PATH}', flush=True)
    print(f'Temporary segments preserved at: {temp_dir}', flush=True)
    # shutil.rmtree(temp_dir)


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--render-segment':
        render_video_segment(int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    else:
        main()
