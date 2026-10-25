"""Окно живого просмотра: показывает игру среды с live=True.

Запуск: uv run python scripts/watch.py [--scale 3] [--screenshot path --frames N]
Окно можно открывать и закрывать в любой момент — обучение от этого не зависит.
"""

import argparse
import time

import numpy as np
import pygame

from neuronoid.live import FRAME_SHAPE, LiveReader, live_name

FPS = 70


def status_line(info: dict, n: int) -> str:
    def show(v):
        return "?" if v is None else v

    return (
        f"счёт {show(info.get('score'))}   раунд {show(info.get('round'))}   "
        f"жизни {show(info.get('lives'))}   кадр {n}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--name", default=live_name())
    ap.add_argument("--screenshot", help="сохранить кадр окна в файл и выйти")
    ap.add_argument("--frames", type=int, default=60, help="через сколько кадров делать скриншот")
    args = ap.parse_args()

    h, w, _ = FRAME_SHAPE
    bar = 24
    pygame.init()
    screen = pygame.display.set_mode((w * args.scale, h * args.scale + bar))
    pygame.display.set_caption("Neuronoid — живой просмотр")
    font = pygame.font.SysFont("consolas,menlo,monospace", 16)
    clock = pygame.time.Clock()
    reader = LiveReader(args.name)
    shown, last_try = 0, 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        got = None
        if time.monotonic() - last_try > 0 or shown:
            got = reader.read()
            if got is None:
                last_try = time.monotonic() + 1.0  # переподключение раз в секунду
        screen.fill((0, 0, 0))
        if got is None:
            text = font.render("ждём обучение…", True, (200, 200, 200))
            screen.blit(text, (10, 10))
        else:
            frame, info, n = got
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(frame.transpose(1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (w * args.scale, h * args.scale)), (0, 0))
            screen.blit(font.render(status_line(info, n), True, (255, 255, 0)), (8, h * args.scale + 3))
            shown += 1
        pygame.display.flip()
        if args.screenshot and shown >= args.frames:
            pygame.image.save(screen, args.screenshot)
            running = False
        clock.tick(FPS)
    reader.close()
    pygame.quit()


if __name__ == "__main__":
    main()
