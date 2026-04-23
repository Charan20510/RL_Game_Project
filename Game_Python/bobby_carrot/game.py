from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import pygame
    from pygame import Surface, Rect
except ImportError:  # pygame not available
    pygame = None
    Surface = None

    class Rect:
        """Minimal fallback when pygame is not installed (headless training)."""
        __slots__ = ("x", "y", "w", "h")

        def __init__(self, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
            self.x = x
            self.y = y
            self.w = w
            self.h = h

        def move(self, dx: int, dy: int) -> "Rect":
            return Rect(self.x + dx, self.y + dy, self.w, self.h)
from enum import Enum

# --- constants ---
FRAMES = 60
FRAMES_PER_STEP = 2
WIDTH_POINTS = 16
HEIGHT_POINTS = 16
VIEW_WIDTH_POINTS = 10
VIEW_HEIGHT_POINTS = 12

MS_PER_FRAME = 1000 // FRAMES
WIDTH = 32 * WIDTH_POINTS
HEIGHT = 32 * HEIGHT_POINTS
VIEW_WIDTH = 32 * VIEW_WIDTH_POINTS
VIEW_HEIGHT = 32 * VIEW_HEIGHT_POINTS
WIDTH_POINTS_DELTA = WIDTH_POINTS - VIEW_WIDTH_POINTS
HEIGHT_POINTS_DELTA = HEIGHT_POINTS - VIEW_HEIGHT_POINTS

# --- utility ---

def asset_path(sub: str) -> Path:
    # for the Python port we bundle a copy of the assets under python/assets
    # (maps, images, audio).  Start from the `python/` directory.
    root = Path(__file__).parent.parent
    return root / "assets" / sub


def load_image(sub: str) -> Surface:
    """Load an image from the assets directory and convert for fast blitting."""
    path = asset_path(sub)
    return pygame.image.load(str(path)).convert_alpha()

# --- map handling ---

class Map:
    def __init__(self, kind: str, number: int):
        self.kind = kind  # "normal" or "egg"
        self.number = number

    def __str__(self) -> str:
        if self.kind == "normal":
            return f"Normal-{self.number:02}"
        else:
            return f"Egg-{self.number:02}"

    def load_map_info(self) -> "MapInfo":
        fname = f"{self.kind}{self.number:02}.blm"
        path = asset_path(f"level/{fname}")
        data = path.read_bytes()[4:]
        start_idx = 0
        carrot_total = 0
        egg_total = 0
        for idx, byte in enumerate(data):
            if byte == 19:
                carrot_total += 1
            elif byte == 45:
                egg_total += 1
            elif byte == 21:
                start_idx = idx
        coord_start = (start_idx % 16, start_idx // 16)
        return MapInfo(data=list(data), coord_start=coord_start,
                       carrot_total=carrot_total, egg_total=egg_total)

    def next(self) -> "Map":
        if self.kind == "normal":
            if self.number < 30:
                return Map("normal", self.number + 1)
            else:
                return Map("egg", 1)
        else:
            if self.number < 20:
                return Map("egg", self.number + 1)
            else:
                return Map("normal", 1)

    def previous(self) -> "Map":
        if self.kind == "normal":
            if self.number <= 1:
                return Map("egg", 20)
            else:
                return Map("normal", self.number - 1)
        else:
            if self.number <= 1:
                return Map("normal", 30)
            else:
                return Map("egg", self.number - 1)

class MapInfo:
    def __init__(self, data: List[int], coord_start: Tuple[int, int],
                 carrot_total: int, egg_total: int):
        self.data = data
        self.coord_start = coord_start
        self.carrot_total = carrot_total
        self.egg_total = egg_total

# --- state ---

class State(Enum):
    Idle = 0
    Death = 1
    FadeIn = 2
    FadeOut = 3
    Left = 4
    Right = 5
    Up = 6
    Down = 7

# --- player ---

class Bobby:
    def __init__(self, start_frame: int, start_time: int,
                 coord_src: Tuple[int, int]) -> None:
        self.state = State.FadeIn
        self.next_state: Optional[State] = None
        self.start_frame = start_frame
        self.start_time = start_time
        self.last_action_time = start_time
        self.coord_src = coord_src
        self.coord_dest = coord_src
        self.carrot_count = 0
        self.egg_count = 0
        self.key_gray = 0
        self.key_yellow = 0
        self.key_red = 0
        self.faded_out = False
        self.dead = False

    def is_walking(self) -> bool:
        return self.coord_src != self.coord_dest

    def is_finished(self, map_info: MapInfo) -> bool:
        if map_info.carrot_total > 0:
            return self.carrot_count == map_info.carrot_total
        else:
            return self.egg_count == map_info.egg_total

    def update_next_state(self, state: State, frame: int) -> None:
        if ((frame - self.start_frame) // FRAMES_PER_STEP > 3
                and self.next_state not in {State.Idle, State.Death,
                                           State.FadeIn, State.FadeOut}):
            self.next_state = state

    def update_state(self, state: State, frame: int, map_data: List[int]) -> None:
        self.start_frame = frame
        self.state = state
        self.update_dest(map_data)

    def update_dest(self, map_data: List[int]) -> None:
        old_dest = self.coord_dest
        # compute tentative destination based on state
        if self.state == State.Left and self.coord_dest[0] > 0:
            self.coord_dest = (self.coord_dest[0]-1, self.coord_dest[1])
        elif self.state == State.Right and self.coord_dest[0] < WIDTH_POINTS-1:
            self.coord_dest = (self.coord_dest[0]+1, self.coord_dest[1])
        elif self.state == State.Up and self.coord_dest[1] > 0:
            self.coord_dest = (self.coord_dest[0], self.coord_dest[1]-1)
        elif self.state == State.Down and self.coord_dest[1] < HEIGHT_POINTS-1:
            self.coord_dest = (self.coord_dest[0], self.coord_dest[1]+1)
        # collision logic
        old_pos = self.coord_src[0] + self.coord_src[1]*16
        new_pos = self.coord_dest[0] + self.coord_dest[1]*16
        old_item = map_data[old_pos]
        new_item = map_data[new_pos]
        forbid = False
        # convert many checks
        if new_item < 18:
            forbid = True
        if new_item == 33 and self.key_gray == 0:
            forbid = True
        if new_item == 35 and self.key_yellow == 0:
            forbid = True
        if new_item == 37 and self.key_red == 0:
            forbid = True
        # arrow / conveyor restrictions
        if new_item == 24 and self.state in {State.Right, State.Down}:
            forbid = True
        if new_item == 25 and self.state in {State.Left, State.Down}:
            forbid = True
        if new_item == 26 and self.state in {State.Left, State.Up}:
            forbid = True
        if new_item == 27 and self.state in {State.Right, State.Up}:
            forbid = True
        if new_item in {28, 40, 41} and self.state in {State.Up, State.Down}:
            forbid = True
        if new_item in {29, 42, 43} and self.state in {State.Left, State.Right}:
            forbid = True
        if new_item == 40 and self.state == State.Right:
            forbid = True
        if new_item == 41 and self.state == State.Left:
            forbid = True
        if new_item == 42 and self.state == State.Down:
            forbid = True
        if new_item == 43 and self.state == State.Up:
            forbid = True
        if new_item == 46:
            forbid = True
        # current item restrictions similar
        if old_item == 24 and self.state in {State.Left, State.Up}:
            forbid = True
        if old_item == 25 and self.state in {State.Right, State.Up}:
            forbid = True
        if old_item == 26 and self.state in {State.Right, State.Down}:
            forbid = True
        if old_item == 27 and self.state in {State.Left, State.Down}:
            forbid = True
        if old_item in {28, 40, 41} and self.state in {State.Up, State.Down}:
            forbid = True
        if old_item in {29, 42, 43} and self.state in {State.Left, State.Right}:
            forbid = True
        if old_item == 40 and self.state == State.Right:
            forbid = True
        if old_item == 41 and self.state == State.Left:
            forbid = True
        if old_item == 42 and self.state == State.Down:
            forbid = True
        if old_item == 43 and self.state == State.Up:
            forbid = True
        if new_item == 31:
            self.next_state = State.Death
        if forbid:
            self.coord_dest = old_dest

    def update_texture_position(self, frame: int, map_data: List[int]) -> Tuple[Rect, Rect]:
        delta_frame = frame - self.start_frame
        is_walking = self.coord_src != self.coord_dest
        step = delta_frame // FRAMES_PER_STEP
        src = Rect(0,0,0,0)
        dest = Rect(0,0,0,0)
        # skeletal translation of Rust logic
        if self.state == State.Idle:
            step_idle = (step // 2) % 3
            src = Rect(36 * step_idle, 0, 36, 50)
            dest = Rect(self.coord_src[0]*32 +16 - (36//2),
                        self.coord_src[1]*32 +16 - (50 - 32//2),
                        36,50)
            return src, dest
        if self.state == State.Death:
            step_death = step // 3
            if step_death > 7:
                step_death = 7
            src = Rect((step_death % 8)*44, 0, 44, 54)
            x0 = self.coord_src[0]*32
            y0 = self.coord_src[1]*32
            x1 = self.coord_dest[0]*32
            y1 = self.coord_dest[1]*32
            x = (x1-x0)//2 + x0
            y = (y1-y0)//2 + y0
            dest = Rect(x + 16 - (44//2), y + 16 - (54 - 32//2), 44, 54)
            if step // 3 >= 12:
                self.dead = True
            return src, dest
        if self.state == State.FadeIn:
            src = Rect((8 - step)*36, 0, 36, 50)
            dest = Rect(self.coord_src[0]*32 +16 - (36//2),
                        self.coord_src[1]*32 +16 - (50 - 32//2),
                        36,50)
            if step >= 8:
                self.start_frame = frame
                self.state = State.Down
            return src, dest
        if self.state == State.FadeOut:
            src = Rect(step*36, 0, 36, 50)
            dest = Rect(self.coord_src[0]*32 +16 - (36//2),
                        self.coord_src[1]*32 +16 - (50 - 32//2),
                        36,50)
            if step >= 8:
                self.faded_out = True
            return src, dest
        # movement states
        # each branch sets src/dest similarly to Rust
        # ... (we'll implement below)
        if self.state == State.Left:
            if is_walking:
                src_x = 36 * ((step + 7) % 8)
                dest_x = (self.coord_src[0]*8 - step)*32//8 + 16 - (36//2)
                dest_y = self.coord_src[1]*32 + 16 - (50 - 32//2)
            else:
                src_x = 36*7
                dest_x = self.coord_src[0]*32 +16 - (36//2)
                dest_y = self.coord_src[1]*32 + 16 - (50 - 32//2)
            src = Rect(src_x, 0, 36, 50)
            dest = Rect(dest_x, dest_y, 36, 50)
        elif self.state == State.Right:
            if is_walking:
                src_x = 36 * ((step + 7) % 8)
                dest_x = (self.coord_src[0]*8 + step)*32//8 + 16 - (36//2)
                dest_y = self.coord_src[1]*32 + 16 - (50 - 32//2)
            else:
                src_x = 36*7
                dest_x = self.coord_src[0]*32 +16 - (36//2)
                dest_y = self.coord_src[1]*32 + 16 - (50 - 32//2)
            src = Rect(src_x, 0, 36, 50)
            dest = Rect(dest_x, dest_y, 36, 50)
        elif self.state == State.Up:
            if is_walking:
                src_x = 36 * ((step + 7) % 8)
                dest_x = self.coord_src[0]*32 +16 - (36//2)
                dest_y = (self.coord_src[1]*8 - step)*32//8 + 16 - (50 - 32//2)
            else:
                src_x = 36*7
                dest_x = self.coord_src[0]*32 +16 - (36//2)
                dest_y = self.coord_src[1]*32 + 16 - (50 - 32//2)
            src = Rect(src_x, 0, 36, 50)
            dest = Rect(dest_x, dest_y, 36, 50)
        elif self.state == State.Down:
            if is_walking:
                src_x = 36 * ((step + 7) % 8)
                dest_x = self.coord_src[0]*32 +16 - (36//2)
                dest_y = (self.coord_src[1]*8 + step)*32//8 + 16 - (50 - 32//2)
            else:
                src_x = 36*7
                dest_x = self.coord_src[0]*32 +16 - (36//2)
                dest_y = self.coord_src[1]*32 + 16 - (50 - 32//2)
            src = Rect(src_x, 0, 36, 50)
            dest = Rect(dest_x, dest_y, 36, 50)
        # post-movement updates
        if step == 6 and is_walking and self.next_state == State.Death:
            self.start_frame = frame
            self.state = State.Death
        elif step == 8 and is_walking:
            old_pos = self.coord_src[0] + self.coord_src[1]*16
            new_pos = self.coord_dest[0] + self.coord_dest[1]*16
            item = map_data[old_pos]
            if item == 24:
                map_data[old_pos] = 25
            elif item == 25:
                map_data[old_pos] = 26
            elif item == 26:
                map_data[old_pos] = 27
            elif item == 27:
                map_data[old_pos] = 24
            elif item == 28:
                map_data[old_pos] = 29
            elif item == 29:
                map_data[old_pos] = 28
            elif item == 30:
                map_data[old_pos] = 31
            elif item == 45:
                map_data[old_pos] = 46
                self.egg_count += 1
            # after handling old_pos, now handle new_pos actions
            new_item2 = map_data[new_pos]
            if new_item2 == 19:
                map_data[new_pos] = 20
                self.carrot_count += 1
            elif new_item2 == 22:
                # red switch: toggle various tiles
                for x in range(WIDTH_POINTS):
                    for y in range(HEIGHT_POINTS):
                        pos = x + y * 16
                        val = map_data[pos]
                        if val == 22:
                            map_data[pos] = 23
                        elif val == 23:
                            map_data[pos] = 22
                        elif val == 24:
                            map_data[pos] = 25
                        elif val == 25:
                            map_data[pos] = 26
                        elif val == 26:
                            map_data[pos] = 27
                        elif val == 27:
                            map_data[pos] = 24
                        elif val == 28:
                            map_data[pos] = 29
                        elif val == 29:
                            map_data[pos] = 28
            elif new_item2 == 31:
                pass
            elif new_item2 == 32:
                map_data[new_pos] = 18
                self.key_gray += 1
            elif new_item2 == 33 and self.key_gray > 0:
                map_data[new_pos] = 18
                self.key_gray -= 1
            elif new_item2 == 34:
                map_data[new_pos] = 18
                self.key_yellow += 1
            elif new_item2 == 35 and self.key_yellow > 0:
                map_data[new_pos] = 18
                self.key_yellow -= 1
            elif new_item2 == 36:
                map_data[new_pos] = 18
                self.key_red += 1
            elif new_item2 == 37 and self.key_red > 0:
                map_data[new_pos] = 18
                self.key_red -= 1
            elif new_item2 == 38:
                for x in range(WIDTH_POINTS):
                    for y in range(HEIGHT_POINTS):
                        pos = x + y * 16
                        val = map_data[pos]
                        if val == 38:
                            map_data[pos] = 39
                        elif val == 39:
                            map_data[pos] = 38
                        elif val == 40:
                            map_data[pos] = 41
                        elif val == 41:
                            map_data[pos] = 40
                        elif val == 42:
                            map_data[pos] = 43
                        elif val == 43:
                            map_data[pos] = 42
            elif new_item2 == 40:
                self.next_state = State.Left
            elif new_item2 == 41:
                self.next_state = State.Right
            elif new_item2 == 42:
                self.next_state = State.Up
            elif new_item2 == 43:
                self.next_state = State.Down
            # end of update of new_pos

            self.coord_src = self.coord_dest
            self.start_frame = frame
            if self.next_state is not None:
                self.update_state(self.next_state, frame, map_data)
                self.next_state = None

        return src, dest

# --- assets ---
class Assets:
    def __init__(self):
        # load textures from the Rust asset directory
        self.bobby_idle = load_image("image/bobby_idle.png")
        self.bobby_death = load_image("image/bobby_death.png")
        self.bobby_fade = load_image("image/bobby_fade.png")
        self.bobby_left = load_image("image/bobby_left.png")
        self.bobby_right = load_image("image/bobby_right.png")
        self.bobby_up = load_image("image/bobby_up.png")
        self.bobby_down = load_image("image/bobby_down.png")
        self.tile_conveyor_left = load_image("image/tile_conveyor_left.png")
        self.tile_conveyor_right = load_image("image/tile_conveyor_right.png")
        self.tile_conveyor_up = load_image("image/tile_conveyor_up.png")
        self.tile_conveyor_down = load_image("image/tile_conveyor_down.png")
        self.tileset = load_image("image/tileset.png")
        self.tile_finish = load_image("image/tile_finish.png")
        self.hud = load_image("image/hud.png")
        self.numbers = load_image("image/numbers.png")
        self.help = load_image("image/help.png")

        # sounds
        pygame.mixer.init()
        base = lambda fname: asset_path(f"audio/{fname}")
        # helper for fallback beep
        def _beep():
            try:
                import winsound
                winsound.Beep(1000, 100)
            except Exception:
                print("\a", end="", flush=True)
        try:
            self.snd_carrot = pygame.mixer.Sound(str(base("carrot.mid")))
        except Exception:
            self.snd_carrot = None
        # background music (looping)
        try:
            pygame.mixer.music.load(str(base("title.mid")))
            pygame.mixer.music.play(-1)
        except Exception:
            # if loading fails, play a beep once
            _beep()
        # store beep helper for use elsewhere
        self._beep = _beep

# --- argument helpers ---

def parse_map_arg(arg: str) -> Map:
    try:
        num = int(arg)
        return Map("normal", num)
    except ValueError:
        pass
    if "-" in arg:
        type_str, num_str = arg.split("-", 1)
        num = int(num_str)
        if type_str.lower() == "normal":
            return Map("normal", num)
        elif type_str.lower() == "egg":
            return Map("egg", num)
    raise ValueError(f"Invalid map: {arg}")


def choose_map_interactive() -> Map:
    print("Select a level to play (examples: 5, normal-3, egg-10).\nPress Enter for normal level 1:")
    choice = input("> ").strip()
    if choice == "":
        return Map("normal", 1)
    return parse_map_arg(choice)

# --- main loop ---

def main():
    if pygame is None:
        print("pygame is not installed or failed to import.\n"
              "Either install pygame or run the Rust version using:")
        print("    python run.py [map]")
        sys.exit(1)

    pygame.init()
    # determine starting map
    if len(sys.argv) > 1:
        map_obj = parse_map_arg(sys.argv[1])
    else:
        map_obj = choose_map_interactive()
    map_info_fresh = map_obj.load_map_info()
    map_info = MapInfo(map_info_fresh.data.copy(), map_info_fresh.coord_start,
                       map_info_fresh.carrot_total, map_info_fresh.egg_total)

    scale = 1.0
    window = pygame.display.set_mode((int(VIEW_WIDTH*scale), int(VIEW_HEIGHT*scale)))
    pygame.display.set_caption(f"Bobby Carrot ({map_obj})")
    clock = pygame.time.Clock()
    assets = Assets()
    frame = 0
    timer_start = pygame.time.get_ticks()
    bobby = Bobby(frame, timer_start, map_info.coord_start)

    show_help = False
    full_view = False

    running = True
    while running:
        now_ms = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                code = event.key
                if code == pygame.K_q:
                    running = False
                elif code == pygame.K_r:
                    map_info = MapInfo(map_info_fresh.data.copy(), map_info_fresh.coord_start,
                                       map_info_fresh.carrot_total, map_info_fresh.egg_total)
                    bobby = Bobby(frame, now_ms, map_info.coord_start)
                elif code == pygame.K_n:
                    map_obj = map_obj.next()
                    pygame.display.set_caption(f"Bobby Carrot ({map_obj})")
                    map_info_fresh = map_obj.load_map_info()
                    map_info = MapInfo(map_info_fresh.data.copy(), map_info_fresh.coord_start,
                                       map_info_fresh.carrot_total, map_info_fresh.egg_total)
                    bobby = Bobby(frame, now_ms, map_info.coord_start)
                elif code == pygame.K_p:
                    map_obj = map_obj.previous()
                    pygame.display.set_caption(f"Bobby Carrot ({map_obj})")
                    map_info_fresh = map_obj.load_map_info()
                    map_info = MapInfo(map_info_fresh.data.copy(), map_info_fresh.coord_start,
                                       map_info_fresh.carrot_total, map_info_fresh.egg_total)
                    bobby = Bobby(frame, now_ms, map_info.coord_start)
                elif code == pygame.K_f:
                    # toggle fullscreen mode rather than viewport size
                    full_view = not full_view
                    if full_view:
                        pygame.display.set_mode((0,0), pygame.FULLSCREEN)
                    else:
                        pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
                elif code in (pygame.K_h, pygame.K_F1):
                    show_help = not show_help
                else:
                    show_help = False

        keys = pygame.key.get_pressed()
        state_opt: Optional[State] = None
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            state_opt = State.Left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            state_opt = State.Right
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            state_opt = State.Up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            state_opt = State.Down
        if state_opt is not None:
            bobby.last_action_time = now_ms
            if not bobby.is_walking():
                bobby.update_state(state_opt, frame, map_info.data)
            else:
                bobby.update_next_state(state_opt, frame)

        # sound triggers: detect changes in carrot/egg count or death
        if bobby.carrot_count != getattr(bobby, '_last_carrots', 0):
            if assets.snd_carrot:
                assets.snd_carrot.play()
            else:
                assets._beep()
        if bobby.egg_count != getattr(bobby, '_last_eggs', 0):
            # no dedicated egg sound; reuse carrot or beep
            if assets.snd_carrot:
                assets.snd_carrot.play()
            else:
                assets._beep()
        if bobby.dead and not getattr(bobby, '_was_dead', False):
            # play a special beep for death
            assets._beep()
        bobby._last_carrots = bobby.carrot_count
        bobby._last_eggs = bobby.egg_count
        bobby._was_dead = bobby.dead

        # check win/death logic
        if bobby.dead:
            map_info = MapInfo(map_info_fresh.data.copy(), map_info_fresh.coord_start,
                               map_info_fresh.carrot_total, map_info_fresh.egg_total)
            bobby = Bobby(frame, now_ms, map_info.coord_start)
        elif bobby.is_finished(map_info) and map_info.data[
                bobby.coord_src[0] + bobby.coord_src[1]*16] == 44:
            if bobby.faded_out:
                map_obj = map_obj.next()
                pygame.display.set_caption(f"Bobby Carrot ({map_obj})")
                map_info_fresh = map_obj.load_map_info()
                map_info = MapInfo(map_info_fresh.data.copy(), map_info_fresh.coord_start,
                                   map_info_fresh.carrot_total, map_info_fresh.egg_total)
                bobby = Bobby(frame, now_ms, map_info.coord_start)
            elif bobby.state != State.FadeOut:
                bobby.start_frame = frame
                bobby.state = State.FadeOut
        elif (now_ms - bobby.last_action_time >= 4000
              and not bobby.is_walking()
              and bobby.state not in {State.Idle, State.Death, State.FadeIn,
                                       State.FadeOut}
              and bobby.next_state is None):
            bobby.start_frame = frame
            bobby.state = State.Idle

        # draw
        assetsurfs = assets  # alias
        screen = pygame.display.get_surface()
        screen.fill((0,0,0))

        # calculate viewport offsets (camera)
        if full_view:
            cam_x = cam_y = 0
            x_right_offset = 0
            y_offset = 0
        else:
            # follow Rust code computing x,y then adjusting bounds
            step = (frame - bobby.start_frame)
            x0 = bobby.coord_src[0] * 32
            y0 = bobby.coord_src[1] * 32
            x1 = bobby.coord_dest[0] * 32
            y1 = bobby.coord_dest[1] * 32
            if bobby.state == State.Death:
                cam_x = (x1 - x0) * 6 // 8 + x0 - (VIEW_WIDTH_POINTS // 2) * 32
                cam_y = (y1 - y0) * 6 // 8 + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
            else:
                cam_x = (x1 - x0) * step // (8 * FRAMES_PER_STEP) + x0 - (VIEW_WIDTH_POINTS // 2) * 32
                cam_y = (y1 - y0) * step // (8 * FRAMES_PER_STEP) + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
            cam_x += 16
            cam_y += 16
            if cam_x < 0:
                cam_x = 0
            if cam_x > WIDTH_POINTS_DELTA * 32:
                cam_x = WIDTH_POINTS_DELTA * 32
            if cam_y < 0:
                cam_y = 0
            if cam_y > HEIGHT_POINTS_DELTA * 32:
                cam_y = HEIGHT_POINTS_DELTA * 32
            x_right_offset = WIDTH_POINTS_DELTA * 32 - cam_x
            y_offset = cam_y
        x_offset = cam_x

        # draw map with camera offset
        for x in range(WIDTH_POINTS):
            for y in range(HEIGHT_POINTS):
                item = map_info.data[x + y * 16]
                texture = assetsurfs.tileset
                if item == 44 and bobby.is_finished(map_info):
                    texture = assetsurfs.tile_finish
                elif item == 40:
                    texture = assetsurfs.tile_conveyor_left
                elif item == 41:
                    texture = assetsurfs.tile_conveyor_right
                elif item == 42:
                    texture = assetsurfs.tile_conveyor_up
                elif item == 43:
                    texture = assetsurfs.tile_conveyor_down
                if (item == 44 and bobby.is_finished(map_info)) or 40 <= item <= 43:
                    src = Rect(32 * ((frame // (FRAMES // 10)) % 4), 0, 32, 32)
                else:
                    src = Rect(32 * (item % 8), 32 * (item // 8), 32, 32)
                dest = Rect(x * 32 - cam_x, y * 32 - cam_y, 32, 32)
                if texture is not None:
                    screen.blit(texture, dest, src)  # type: ignore[arg-type]

        # bobby
        bobby_src, bobby_dest = bobby.update_texture_position(frame, map_info.data)
        bobby_tex = {
            State.Idle: assetsurfs.bobby_idle,
            State.Death: assetsurfs.bobby_death,
            State.FadeIn: assetsurfs.bobby_fade,
            State.FadeOut: assetsurfs.bobby_fade,
            State.Left: assetsurfs.bobby_left,
            State.Right: assetsurfs.bobby_right,
            State.Up: assetsurfs.bobby_up,
            State.Down: assetsurfs.bobby_down,
        }[bobby.state]
        # adjust destination by camera
        bobby_dest = bobby_dest.move(-cam_x, -cam_y)
        if bobby_tex is not None:
            screen.blit(bobby_tex, bobby_dest, bobby_src)  # type: ignore[arg-type]

        # HUD indicator
        if map_info.carrot_total > 0:
            icon_rect = Rect(0, 0, 46, 44)
            num_left = map_info.carrot_total - bobby.carrot_count
            icon_width = 46
        else:
            icon_rect = Rect(46, 0, 34, 44)
            num_left = map_info.egg_total - bobby.egg_count
            icon_width = 34
        if assetsurfs.hud is not None:
            screen.blit(assetsurfs.hud, (32 * 16 - (icon_width + 4) - x_right_offset,
                                          4 + y_offset), icon_rect)  # type: ignore[arg-type]
        if assetsurfs.numbers is not None:
            num_10 = num_left // 10
            num_01 = num_left % 10
            screen.blit(assetsurfs.numbers,
                        (32 * 16 - (icon_width + 4) - 2 - 12 - x_right_offset,
                         4 + 14 + y_offset),
                        Rect(num_01 * 12, 0, 12, 18))  # type: ignore[arg-type]
            screen.blit(assetsurfs.numbers,
                        (32 * 16 - (icon_width + 4) - 2 - 12 * 2 - 1 - x_right_offset,
                         4 + 14 + y_offset),
                        Rect(num_10 * 12, 0, 12, 18))  # type: ignore[arg-type]
        # keys
        keys = []
        for _ in range(bobby.key_gray):
            keys.append((122, len(keys)))
        for _ in range(bobby.key_yellow):
            keys.append((122 + 22, len(keys)))
        for _ in range(bobby.key_red):
            keys.append((122 + 22 + 22, len(keys)))
        if assetsurfs.hud is not None:
            for offset, count in keys:
                screen.blit(assetsurfs.hud,
                            (32 * 16 - (22 + 4) - count * 22 - x_right_offset,
                             4 + 44 + 2 + y_offset),
                            Rect(offset, 0, 22, 44))  # type: ignore[arg-type]
        # time passed
        passed_secs = (now_ms - bobby.start_time) // 1000
        minutes = passed_secs // 60
        seconds = passed_secs % 60
        if minutes > 99:
            minutes = 99
            seconds = 99
        if assetsurfs.numbers is not None:
            for idx, offset in enumerate([minutes // 10, minutes % 10, 10,
                                           seconds // 10, seconds % 10]):
                screen.blit(assetsurfs.numbers,
                            (4 + 12 * idx + x_offset, 4 + y_offset),
                            Rect(offset * 12, 0, 12, 18))  # type: ignore[arg-type]
        # help page
        if show_help:
            s = pygame.Surface((158, 160), pygame.SRCALPHA)
            s.fill((0, 0, 0, 200))
            if assetsurfs.help is not None:
                screen.blit(s, ((32 * 16 - cam_x - x_right_offset - 158) // 2 + cam_x,
                                 32 * 3 - (160 - 142) // 2 + y_offset))
                screen.blit(assetsurfs.help,
                            ((32 * 16 - cam_x - x_right_offset - 133) // 2 + cam_x,
                             32 * 3 + y_offset),
                            Rect(0, 0, 133, 142))  # type: ignore[arg-type]

        pygame.display.flip()
        frame += 1
        clock.tick(FRAMES)

    pygame.quit()


if __name__ == "__main__":
    main()
